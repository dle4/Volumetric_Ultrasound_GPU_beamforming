import numpy as np
import os
import torch # Import PyTorch
import time # Import time module
import torch.profiler
import matplotlib.pyplot as plt # Import matplotlib for visualization

# Import the compiled PyTorch C++ extension
import fused_beamform_ext

# Import the original vectorized beamform function for comparison
from vectorized_beamformer3D import vectorized_beamform as original_vectorized_beamform

# Remove the JIT-scripted function as its logic is now in the fused kernel
# @torch.jit.script
# def _process_chunk(...):
#     ... (removed)

def vectorized_beamform_ext(iq_data: torch.Tensor, xi: torch.Tensor, yi: torch.Tensor, zi: torch.Tensor, txdel: torch.Tensor, element_pos: torch.Tensor, fs: float, fc: float, num_elements: int, c: float = 1540.0, device: str = 'cuda', z_chunk_size: int = 8, x_chunk_size: int = 8, y_chunk_size: int = 16):
    """
    Performs vectorized Delay-and-Sum beamforming on I/Q data using a fused CUDA kernel
    implemented as a PyTorch C++ Extension, with chunking to manage memory.
    Accepts PyTorch Tensors as input, including element positions.

    Args:
        iq_data (torch.Tensor): Input I/Q data. Shape: (batch_size, num_samples, num_elements) or (num_samples, num_elements)
                                             Must be a PyTorch Tensor on the specified device.
        xi (torch.Tensor): X-coordinates of the beamforming grid. Shape: (Nx, Ny, Nz)
        yi (torch.Tensor): Y-coordinates of the beamforming grid. Shape: (Nx, Ny, Nz)
        zi (torch.Tensor): Z-coordinates of the beamforming grid. Shape: (Nx, Ny, Nz)
        txdel (torch.Tensor): Transmit delays for each element. Shape: (1, num_elements) or (num_tx_events, num_elements)
                                          (Assuming single transmit event for now based on simulate_data.py)
        element_pos (torch.Tensor): Transducer element positions. Shape: (2, num_elements) -> [x_coords, y_coords]. Must be a PyTorch Tensor on the specified device.
        fs (float): Sampling frequency in Hz.
        fc (float): Center frequency for I/Q phase rotation in Hz.
        num_elements (int): Number of transducer elements.
        c (float, optional): Speed of sound in m/s. Defaults to 1540.0.
        device (str, optional): The device to perform computation on ('cpu' or 'cuda'). Defaults to 'cuda'.
        z_chunk_size (int, optional): The number of Z-slices to process in each chunk. Defaults to 8.
        x_chunk_size (int, optional): The number of X-points to process in each chunk. Defaults to 8.
        y_chunk_size (int, optional): The number of Y-points to process in each chunk. Defaults to 16.
    Returns:
        torch.Tensor: Beamformed I/Q data. Shape: (batch_size, Nx, Ny, Nz) or (Nx, Ny, Nz)
                                    Returns PyTorch Tensor on the specified device.
    """
    print(f"Starting fused vectorized beamforming (PyTorch C++ Extension) on device: {device}...")

    # --- Input Validation and Parameter Extraction ---
    if not (isinstance(iq_data, torch.Tensor) and isinstance(xi, torch.Tensor) and
            isinstance(yi, torch.Tensor) and isinstance(zi, torch.Tensor) and
            isinstance(txdel, torch.Tensor) and isinstance(element_pos, torch.Tensor)):
        raise TypeError("All inputs (iq_data, xi, yi, zi, txdel, element_pos) must be PyTorch Tensors.")

    if iq_data.ndim == 3:
        batch_size, num_samples, num_elements_iq = iq_data.shape
        # For the fused kernel, we process one batch item at a time.
        # The kernel expects [elements][samples] layout for IQ data.
        # We will transpose and process each batch item in the loop.
    elif iq_data.ndim == 2:
        num_samples, num_elements_iq = iq_data.shape
        batch_size = 1
        iq_data = iq_data.unsqueeze(0) # Add batch dim for consistent loop
    else:
        raise ValueError("iq_data must have 2 or 3 dimensions (num_samples, num_elements) or (batch_size, num_samples, num_elements)")

    if num_elements_iq != num_elements:
        raise ValueError(f"Mismatch between number of elements in iq_data ({num_elements_iq}) and provided num_elements ({num_elements})")

    # --- Process Input Element Coordinates ---
    # element_pos shape: (2, num_elements) -> [x_coords, y_coords]
    if element_pos.shape != (2, num_elements):
         raise ValueError(f"element_pos must have shape (2, num_elements), but got {element_pos.shape}")

    # The kernel expects 1D tensors for element coordinates, flattened in C-order.
    # Assuming element_pos is already ordered correctly or can be flattened directly.
    # If element_pos comes from param_data['elements'], it's likely (2, num_elements)
    # where num_elements is the total count, flattened from a 2D array in C-order.
    # We need to ensure the x, y, z tensors passed to the kernel are 1D and match this order.

    # Extract x and y coordinates and ensure they are on the correct device and dtype
    element_x_tensor = element_pos[0, :].to(torch.float32).to(device)
    element_y_tensor = element_pos[1, :].to(torch.float32).to(device)
    # Create a tensor of zeros for the z-coordinates, matching the number of elements
    element_z_tensor = torch.zeros_like(element_x_tensor, dtype=torch.float32, device=device)

    # Ensure the 1D element tensors have the correct number of elements
    if element_x_tensor.shape[0] != num_elements or element_y_tensor.shape[0] != num_elements or element_z_tensor.shape[0] != num_elements:
         raise ValueError(f"Mismatch in processed element coordinate tensor sizes. Expected {num_elements}, got x:{element_x_tensor.shape[0]}, y:{element_y_tensor.shape[0]}, z:{element_z_tensor.shape[0]}")


    if txdel.shape[1] != num_elements:
         raise ValueError(f"Mismatch between number of elements in txdel ({txdel.shape[1]}) and provided num_elements ({num_elements})")
    # Assuming single transmit event for now, matching simulate_data.py
    if txdel.shape[0] != 1:
        print("Warning: txdel has more than one row, assuming single transmit event and using the first row.")
        txdel = txdel[0:1, :] # Use only the first row
    txdel_tensor = txdel[0, :].to(torch.float32).to(device) # Kernel expects 1D tensor

    grid_shape = xi.shape # (Nx, Ny, Nz)
    Nx, Ny, Nz = grid_shape
    # Keep original xi_np for plotting extent
    # xi_np = xi.cpu().numpy() # Already loaded as numpy

    # Determine chunking for X and Y dimensions
    if x_chunk_size == -1 or x_chunk_size > Nx:
        x_chunk_size = Nx
    if y_chunk_size == -1 or y_chunk_size > Ny:
        y_chunk_size = Ny

    # Initialize the output tensor on the specified device (complex64)
    beamformed_iq = torch.zeros((batch_size, Nx, Ny, Nz), dtype=torch.complex64, device=device)

    # --- Chunking along Z-axis ---
    z_chunk_size = min(z_chunk_size, Nz) # Ensure z_chunk_size does not exceed Nz

    # --- Nested Chunking along X and Y axes ---
    print(f"Processing grid in chunks of size ({x_chunk_size}, {y_chunk_size}, {z_chunk_size}) along X, Y, Z axes...")

    # Iterate through batches
    for b in range(batch_size):
        iq_data_batch = iq_data[b, :, :].to(device) # Select current batch item
        # Transpose IQ data for the kernel: [samples][elements] -> [elements][samples]
        iq_data_transposed = iq_data_batch.transpose(0, 1).contiguous() # Ensure contiguity

        z_start = 0
        while z_start < Nz:
            z_end = min(z_start + z_chunk_size, Nz)
            current_Nz = z_end - z_start
            if current_Nz == 0:
                break # Should not happen if z_start < Nz, but as a safeguard

            x_start = 0
            while x_start < Nx:
                x_end = min(x_start + x_chunk_size, Nx)
                current_Nx = x_end - x_start
                if current_Nx == 0:
                    break # Should not happen if x_start < Nx

                y_start = 0
                while y_start < Ny:
                    y_end = min(y_start + y_chunk_size, Ny)
                    current_Ny = y_end - y_start
                    if current_Ny == 0:
                        break # Should not happen if y_start < Ny

                    # Select and flatten grid points for the current XY-Z chunk
                    xi_chunk = xi[x_start:x_end, y_start:y_end, z_start:z_end].ravel().to(torch.float32).to(device).contiguous()
                    yi_chunk = yi[x_start:x_end, y_start:y_end, z_start:z_end].ravel().to(torch.float32).to(device).contiguous()
                    zi_chunk = zi[x_start:x_end, y_start:y_end, z_start:z_end].ravel().to(torch.float32).to(device).contiguous()
                    num_chunk_points = current_Nx * current_Ny * current_Nz

                    # Call the fused kernel from the C++ extension
                    # The extension function expects flattened grid points and 1D element/txdel tensors
                    beamformed_iq_chunk_flat = fused_beamform_ext.fused_beamform(
                        iq_data_transposed, # [elements][samples]
                        xi_chunk,           # flattened chunk x
                        yi_chunk,           # flattened chunk y
                        zi_chunk,           # flattened chunk z
                        element_x_tensor,   # 1D element x (C-order)
                        element_y_tensor,   # 1D element y (C-order)
                        element_z_tensor,   # 1D element z (C-order)
                        txdel_tensor,       # 1D tx delays
                        fs,
                        fc,
                        float(c)
                    )

                    # Reshape the flat chunk data back to its original chunk dimensions
                    beamformed_iq_chunk_reshaped = beamformed_iq_chunk_flat.reshape((current_Nx, current_Ny, current_Nz))

                    # Store the beamformed chunk in the final output tensor
                    beamformed_iq[b, x_start:x_end, y_start:y_end, z_start:z_end] = beamformed_iq_chunk_reshaped

                    y_start += y_chunk_size
                x_start += x_chunk_size
            z_start += z_chunk_size


    # --- Return the final beamformed volume ---
    return beamformed_iq # Always return PyTorch Tensor from this function


# Example Usage (requires saved data from simulate_data.py)
if __name__ == '__main__':
    print("\n--- Running Example Usage (PyTorch C++ Extension) ---")
    # --- Environment Setup & Verification (Step 1) ---
    print("\n--- Environment Setup & Verification ---")
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("CUDA is not available. Cannot run CUDA example usage.")
        exit() # Exit if CUDA is not available
    print(f"Selected device: {device}")
    # --- End Environment Setup & Verification ---

    data_dir = "simulated_data"
    try:
        # Load data (as NumPy arrays)
        print(f"Loading data from '{data_dir}'...")
        xi_np = np.load(os.path.join(data_dir, 'xi.npy'))
        yi_np = np.load(os.path.join(data_dir, 'yi.npy'))
        zi_np = np.load(os.path.join(data_dir, 'zi.npy'))
        txdel3_np = np.load(os.path.join(data_dir, 'txdel3.npy'))
        param_data = np.load(os.path.join(data_dir, 'param_data.npy'), allow_pickle=True).item()
        print("Data loaded successfully.")

        # Demodulate RF data (as per plan, this is done outside the beamformer)
        print("Demodulating RF data...")
        import pymust # Import pymust here for the example usage
        # Need to load RF data to demodulate
        RF3_np = np.load(os.path.join(data_dir, 'RF3.npy'))
        IQ3_np = pymust.rf2iq(RF3_np, param_data['fs'], param_data['fc'])
        print(f"I/Q data generated. Shape: {IQ3_np.shape}")

        # Extract parameters and convert all inputs to tensors and move to device before calling beamform function
        fs = param_data['fs']
        fc = param_data['fc']
        num_elements = param_data['Nelements']
        element_pos_np = param_data['elements'] # Load original element_pos
        speed_of_sound = 1540.0 # Use default c

        # Convert NumPy arrays to PyTorch tensors and move to device
        # Add batch dimension to IQ data for consistency with the function signature
        IQ3_tensor = torch.from_numpy(IQ3_np).to(torch.complex64).unsqueeze(0).to(device) # Shape: (1, num_samples, num_elements)
        xi_tensor = torch.from_numpy(xi_np).to(torch.float32).to(device)
        yi_tensor = torch.from_numpy(yi_np).to(torch.float32).to(device)
        zi_tensor = torch.from_numpy(zi_np).to(torch.float32).to(device)
        txdel3_tensor = torch.from_numpy(txdel3_np).to(torch.float32).to(device) # Shape: (1, num_elements)
        element_pos_tensor = torch.from_numpy(element_pos_np).to(torch.float32).to(device) # Convert element_pos to tensor

        # Determine batch size and num_samples from the IQ tensor
        batch_size, num_samples, _ = IQ3_tensor.shape


        # Flag to enable/disable profiling
        enable_profiling = False # Set to True to enable profiling

        # --- Run both implementations for comparison ---

        # Run original vectorized beamformer (PyTorch JIT)
        print("\n--- Running original vectorized_beamform (PyTorch JIT) ---")
        original_start_time = time.time()

        bIQ3_original_tensor = original_vectorized_beamform(
             IQ3_tensor, xi_tensor, yi_tensor, zi_tensor, txdel3_tensor, element_pos_tensor, # Pass element_pos_tensor
             fs, fc, num_elements, c=speed_of_sound, device=device
        )
        original_end_time = time.time()
        original_elapsed_time = original_end_time - original_start_time
        print(f"Original Beamforming time on CUDA (PyTorch JIT): {original_elapsed_time:.4f} seconds per volume")


        # Run fused vectorized beamformer (PyTorch C++ Extension)
        print("\n--- Running vectorized_beamform_ext with optimal chunking and C++ Extension ---")
        fused_start_event = torch.cuda.Event(enable_timing=True)
        fused_end_event = torch.cuda.Event(enable_timing=True)

        if enable_profiling:
            # Profiler start
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                # Code to be profiled
                fused_start_event.record()
                bIQ3_fused_tensor = vectorized_beamform_ext(
                    IQ3_tensor, xi_tensor, yi_tensor, zi_tensor, txdel3_tensor, element_pos_tensor, # Pass element_pos_tensor
                    fs, fc, num_elements, c=speed_of_sound, device=device
                )
                fused_end_event.record()
                torch.cuda.synchronize() # Wait for events to complete
                prof.step() # Mark the end of a step in profiling
        else:
             # Call the main beamform function without profiling
             fused_start_event.record()
             bIQ3_fused_tensor = vectorized_beamform_ext(
                IQ3_tensor, xi_tensor, yi_tensor, zi_tensor, txdel3_tensor, element_pos_tensor, # Pass element_pos_tensor
                fs, fc, num_elements, c=speed_of_sound, device=device
             )
             fused_end_event.record()
             torch.cuda.synchronize() # Wait for events to complete


        # Calculate elapsed time for fused kernel
        fused_elapsed_time_ms = fused_start_event.elapsed_time(fused_end_event)
        print(f"Fused Beamforming time on CUDA (C++ Extension): {fused_elapsed_time_ms:.4f} ms")


        # --- Visual Comparison ---
        print("\n--- Generating Visual Comparison ---")
        output_dir = "example_outputs"
        os.makedirs(output_dir, exist_ok=True)

        # Select a slice for visualization (e.g., a Z-slice near the scatterer)
        # The scatterer is at z = 30e-3. Find the closest Z index.
        z_coords = zi_np[0, 0, :] # Assuming zi_np is (Nx, Ny, Nz)
        scatterer_z_coord = 30e-3
        closest_z_index = np.argmin(np.abs(z_coords - scatterer_z_coord))

        # Extract the selected Z-slice from both outputs (remove batch dim)
        original_slice_np = bIQ3_original_tensor.squeeze(0)[:, :, closest_z_index].abs().cpu().numpy()
        fused_slice_np = bIQ3_fused_tensor.squeeze(0)[:, :, closest_z_index].abs().cpu().numpy()

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Display the original slice
        im1 = axes[0].imshow(original_slice_np.T, aspect='auto', cmap='gray',
                             extent=[xi_np.min(), xi_np.max(), zi_np.min(), zi_np.max()]) # Use xi and zi ranges for extent
        axes[0].set_title('Original PyTorch JIT Output')
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Z (m)')
        fig.colorbar(im1, ax=axes[0])

        # Display the fused slice
        im2 = axes[1].imshow(fused_slice_np.T, aspect='auto', cmap='gray',
                             extent=[xi_np.min(), xi_np.max(), zi_np.min(), zi_np.max()]) # Use xi and zi ranges for extent
        axes[1].set_title('Fused C++ Extension Output')
        axes[1].set_xlabel('X (m)')
        axes[1].set_ylabel('Z (m)')
        fig.colorbar(im2, ax=axes[1])

        # Add a title for the entire figure
        fig.suptitle(f'Beamformed Output Comparison (Z-slice near {scatterer_z_coord*1000:.2f} mm)')

        # Save the figure
        output_filename = os.path.join(output_dir, 'beamformed_output_comparison.png')
        plt.savefig(output_filename)
        print(f"Visual comparison saved to '{output_filename}'")

        # Close the plot to prevent it from displaying immediately
        plt.close(fig)


    except FileNotFoundError:
        print(f"Error: Data files not found in '{data_dir}'. Please run simulate_data.py first.")
    except ImportError:
         print("Error: Required libraries (numpy, torch, pymust, fused_beamform_ext, matplotlib) not found. Please install them.")
    except Exception as e:
        print(f"An error occurred during example usage: {e}")
        import traceback
        traceback.print_exc()