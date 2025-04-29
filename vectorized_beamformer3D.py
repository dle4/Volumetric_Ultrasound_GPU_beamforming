import numpy as np
import os
import torch # Import PyTorch
import time # Import time module
import torch.profiler # Import torch.profiler
import torch.jit

from torch.cuda.amp import autocast

@torch.jit.script
def _process_chunk(
    iq_tensor_chunk: torch.Tensor,
    xi_chunk: torch.Tensor,
    yi_chunk: torch.Tensor,
    zi_chunk: torch.Tensor,
    txdel_tensor: torch.Tensor,
    element_coords_3d: torch.Tensor,
    fs: float,
    fc: float,
    c: float,
    num_samples: int,
    num_elements: int,
    batch_size: int
) -> torch.Tensor:
    """
    Processes a single chunk of the beamforming grid. This function is JIT-scripted.
    """
    wc = 2 * np.pi * fc
    current_Nx, current_Ny, current_Nz = xi_chunk.shape
    num_chunk_points = current_Nx * current_Ny * current_Nz

    # Reshape chunk grid points for broadcasting: (current_Nx*current_Ny*current_Nz, 1, 3)
    grid_points_chunk = torch.stack([xi_chunk.ravel(), yi_chunk.ravel(), zi_chunk.ravel()], dim=-1).unsqueeze(1)

    # --- Delay Calculation (Vectorized with PyTorch) ---
    # Calculate receive distance: distance from each element to each grid point in chunk
    # Broadcasting: (current_Nx*current_Ny*current_Nz, 1, 3) - (1, num_elements, 3) -> (current_Nx*current_Ny*current_Nz, num_elements, 3)
    receive_dist_chunk = torch.sqrt(torch.sum((grid_points_chunk - element_coords_3d)**2, dim=2)) # Shape: (current_Nx*current_Ny*current_Nz, num_elements)

    # Calculate transmit distance (matching dasmtx3 logic for passive=False, useVirtualSource=False)
    # dTX = np.min(delaysTX*c + dRX, axis = 1)
    # Need to broadcast txdel_tensor (1, num_elements) to (current_Nx*current_Ny*current_Nz, num_elements)
    txdel_broadcast_chunk = txdel_tensor.expand(num_chunk_points, -1) # Shape: (current_Nx*current_Ny*current_Nz, num_elements)
    dTX_chunk = torch.min(txdel_broadcast_chunk * c + receive_dist_chunk, dim=1, keepdim=True).values # Shape: (current_Nx*current_Ny*current_Nz, 1)

    # Calculate total time delay = transmit delay + receive delay
    # Transmit delay shape: (current_Nx*current_Ny*current_Nz, 1)
    # Receive delay shape: (current_Nx*current_Ny*current_Nz, num_elements)
    # Broadcasting: (current_Nx*current_Ny*current_Nz, 1) + (current_Nx*current_Ny*current_Nz, num_elements) -> (current_Nx*current_Ny*current_Nz, num_elements)
    total_delay_chunk = (dTX_chunk / c) + (receive_dist_chunk / c) # Shape: (current_Nx*current_Ny*current_Nz, num_elements)

    # Convert delays to sample indices
    sample_indices_chunk = total_delay_chunk * fs # Shape: (current_Nx*current_Ny*current_Nz, num_elements)

    # --- Interpolation (Linear) and Apodization (Vectorized with PyTorch) ---
    # Linear interpolation using gather
    # sample_indices_chunk shape: (current_Nx*current_Ny*current_Nz, num_elements)
    i_chunk = torch.floor(sample_indices_chunk).long() # Integer part of sample index
    i_plus_1_chunk = i_chunk + 1 # Next integer sample index

    # Clamp indices to valid range [0, num_samples - 1]
    i_clamped_chunk = torch.clamp(i_chunk, 0, num_samples - 1)
    i_plus_1_clamped_chunk = torch.clamp(i_plus_1_chunk, 0, num_samples - 1)

    # Get the weights for linear interpolation
    w1_chunk = (i_plus_1_chunk.float() - sample_indices_chunk) # Weight for sample at i
    w2_chunk = (sample_indices_chunk - i_chunk.float())       # Weight for sample at i + 1

    # Handle out-of-bounds indices by setting weights to 0
    valid_mask_i_chunk = (i_chunk >= 0) & (i_chunk < num_samples)
    valid_mask_i_plus_1_chunk = (i_plus_1_chunk >= 0) & (i_plus_1_chunk < num_samples)

    w1_chunk = w1_chunk * valid_mask_i_chunk.float()
    w2_chunk = w2_chunk * valid_mask_i_plus_1_chunk.float()

    # Reshape iq_tensor for gathering: (batch_size, num_elements, num_samples)
    iq_tensor_transposed = iq_tensor_chunk.transpose(1, 2)

    # Reshape clamped indices to (1, num_elements, current_Nx*current_Ny*current_Nz) for broadcasting with iq_tensor_transposed
    i_clamped_reshaped_chunk = i_clamped_chunk.unsqueeze(0).transpose(1, 2)
    i_plus_1_clamped_reshaped_chunk = i_plus_1_clamped_chunk.unsqueeze(0).transpose(1, 2)

    # Gather samples: (batch_size, num_elements, current_Nx*current_Ny*current_Nz)
    gathered_samples_i_chunk = torch.gather(iq_tensor_transposed, 2, i_clamped_reshaped_chunk.expand(batch_size, -1, -1))
    gathered_samples_i_plus_1_chunk = torch.gather(iq_tensor_transposed, 2, i_plus_1_clamped_reshaped_chunk.expand(batch_size, -1, -1))

    # Reshape weights for broadcasting: (1, num_elements, current_Nx*current_Ny*current_Nz)
    w1_reshaped_chunk = w1_chunk.unsqueeze(0).transpose(1, 2)
    w2_reshaped_chunk = w2_chunk.unsqueeze(0).transpose(1, 2)

    # Perform linear interpolation
    interpolated_samples_chunk = gathered_samples_i_chunk * w1_reshaped_chunk + gathered_samples_i_plus_1_chunk * w2_reshaped_chunk # Shape: (batch_size, num_elements, current_Nx*current_Ny*current_Nz)

    # --- Apodization ---
    # Simple rectangular apodization (weights are 1 where valid, 0 otherwise)
    # The valid_mask_i_chunk already captures the valid range for interpolation.
    apodization_weights_chunk = valid_mask_i_chunk.unsqueeze(0).transpose(1, 2).float() # Shape: (batch_size, num_elements, current_Nx*current_Ny*current_Nz)

    # Apply apodization
    apodized_samples_chunk = interpolated_samples_chunk * apodization_weights_chunk # Shape: (batch_size, num_elements, current_Nx*current_Ny*current_Nz)

    # --- I/Q Phase Rotation ---
    # Apply phase rotation: exp(1j * wc * total_delay)
    # total_delay_chunk shape: (current_Nx*current_Ny*current_Nz, num_elements)
    # Reshape total_delay_chunk for broadcasting: (1, num_elements, current_Nx*current_Ny*current_Nz)
    total_delay_reshaped_chunk = total_delay_chunk.unsqueeze(0).transpose(1, 2)
    phase_rotation_chunk = torch.exp(torch.complex(torch.zeros_like(total_delay_reshaped_chunk), wc * total_delay_reshaped_chunk)) # Shape: (1, num_elements, current_Nx*current_Ny*current_Nz)

    # Apply phase rotation to apodized samples
    rotated_samples_chunk = apodized_samples_chunk * phase_rotation_chunk # Shape: (batch_size, num_elements, current_Nx*current_Ny*current_Nz)

    # --- Summation ---
    beamformed_iq_flat_chunk = torch.sum(rotated_samples_chunk, dim=1) # Shape: (batch_size, current_Nx*current_Ny*current_Nz)

    # Reshape to match the chunk grid dimensions
    beamformed_iq_chunk = beamformed_iq_flat_chunk.reshape((batch_size, current_Nx, current_Ny, current_Nz)) # Shape: (batch_size, current_Nx, current_Ny, current_Nz)

    return beamformed_iq_chunk


def vectorized_beamform(iq_data: torch.Tensor, xi: torch.Tensor, yi: torch.Tensor, zi: torch.Tensor, txdel: torch.Tensor, element_pos: torch.Tensor, fs: float, fc: float, num_elements: int, c: float = 1540.0, device: str = 'cuda', z_chunk_size: int = 8, x_chunk_size: int = 8, y_chunk_size: int = 16):
    """
    Performs vectorized Delay-and-Sum beamforming on I/Q data using PyTorch,
    with chunking along the Z-axis and optionally along X and Y axes to manage memory.
    Calls a JIT-scripted function for chunk processing.
    Accepts PyTorch Tensors as input.

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
    print(f"Starting vectorized beamforming (PyTorch implementation) on device: {device} with JIT...")

    # --- Input Validation and Parameter Extraction ---
    if not (isinstance(iq_data, torch.Tensor) and isinstance(xi, torch.Tensor) and
            isinstance(yi, torch.Tensor) and isinstance(zi, torch.Tensor) and
            isinstance(txdel, torch.Tensor) and isinstance(element_pos, torch.Tensor)):
        raise TypeError("All inputs (iq_data, xi, yi, zi, txdel, element_pos) must be PyTorch Tensors.")

    if iq_data.ndim == 3:
        batch_size, num_samples, num_elements_iq = iq_data.shape
    elif iq_data.ndim == 2:
        num_samples, num_elements_iq = iq_data.shape
        batch_size = 1
        iq_data = iq_data.unsqueeze(0) # Add batch dim
    else:
        raise ValueError("iq_data must have 2 or 3 dimensions (num_samples, num_elements) or (batch_size, num_samples, num_elements)")

    if num_elements_iq != num_elements:
        raise ValueError(f"Mismatch between number of elements in iq_data ({num_elements_iq}) and provided num_elements ({num_elements})")

    element_coords_3d = torch.zeros((1, num_elements, 3), dtype=torch.float32, device=device)
    element_coords_3d[0, :, 0] = element_pos[0, :].to(torch.float32) # x-coords
    element_coords_3d[0, :, 1] = element_pos[1, :].to(torch.float32) # y-coords


    if txdel.shape[1] != num_elements:
         raise ValueError(f"Mismatch between number of elements in txdel ({txdel.shape[1]}) and provided num_elements ({num_elements})")
    # Assuming single transmit event for now, matching simulate_data.py
    if txdel.shape[0] != 1:
        print("Warning: txdel has more than one row, assuming single transmit event and using the first row.")
        txdel = txdel[0:1, :] # Use only the first row

    grid_shape = xi.shape # (Nx, Ny, Nz)
    Nx, Ny, Nz = grid_shape

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
    print(f"Processing grid in chunks of size ({x_chunk_size}, {y_chunk_size}) along X and Y axes within each Z chunk...")

    # Iterate through chunks
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

                # Select grid points for the current XY-Z chunk
                xi_chunk = xi[x_start:x_end, y_start:y_end, z_start:z_end]
                yi_chunk = yi[x_start:x_end, y_start:y_end, z_start:z_end]
                zi_chunk = zi[x_start:x_end, y_start:y_end, z_start:z_end]

                # Call the JIT-scripted function to process the chunk
                beamformed_iq_chunk = _process_chunk(
                    iq_data, # Pass the full iq_data tensor
                    xi_chunk,
                    yi_chunk,
                    zi_chunk,
                    txdel, # Pass the full txdel tensor
                    element_coords_3d, # Pass the element_coords_3d tensor
                    fs,
                    fc,
                    float(c), # Cast c to float for JIT
                    num_samples,
                    num_elements,
                    batch_size
                )

                # Store the beamformed chunk in the final output tensor
                beamformed_iq[:, x_start:x_end, y_start:y_end, z_start:z_end] = beamformed_iq_chunk

                y_start += y_chunk_size
            x_start += x_chunk_size
        z_start += z_chunk_size


    # --- Return the final beamformed volume ---
    # Note: The conversion to numpy is done outside this function.
    return beamformed_iq # Always return PyTorch Tensor from this function


# Example Usage (requires saved data from simulate_data.py)
if __name__ == '__main__':
    print("\n--- Running Example Usage ---")
    # --- Environment Setup & Verification (Step 1) ---
    print("\n--- Environment Setup & Verification ---")
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("CUDA is not available. Using CPU.")
        device = 'cpu'
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
        element_pos_np = param_data['elements']
        speed_of_sound = 1540.0 # Use default c

        IQ3_tensor = torch.from_numpy(IQ3_np).to(torch.complex64).to(device)
        xi_tensor = torch.from_numpy(xi_np).to(torch.float32).to(device)
        yi_tensor = torch.from_numpy(yi_np).to(torch.float32).to(device)
        zi_tensor = torch.from_numpy(zi_np).to(torch.float32).to(device)
        txdel3_tensor = torch.from_numpy(txdel3_np).to(torch.float32).to(device)
        element_pos_tensor = torch.from_numpy(element_pos_np).to(torch.float32).to(device)

        # Determine batch size and num_samples from the IQ tensor
        if IQ3_tensor.ndim == 3:
            batch_size, num_samples, _ = IQ3_tensor.shape
        elif IQ3_tensor.ndim == 2:
            num_samples, _ = IQ3_tensor.shape
            batch_size = 1 # Assuming single batch if no batch dim

        # Flag to enable/disable profiling
        enable_profiling = False # Set to True to enable profiling

        # Run vectorized beamformer with optimal chunking and JIT
        print("\n--- Running vectorized_beamform with optimal chunking and JIT ---")

        if device == 'cuda':
            # Existing timing code
            torch.cuda.synchronize()
            start_time = time.time()

            if enable_profiling:
                # Profiler start
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'), # Corrected handler access
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
                ) as prof:
                    # Code to be profiled
                    # Call the main beamform function with tensors and default optimal chunk sizes
                    bIQ3_vec_tensor = vectorized_beamform(
                        IQ3_tensor, xi_tensor, yi_tensor, zi_tensor, txdel3_tensor, element_pos_tensor,
                        fs, fc, num_elements, c=speed_of_sound, device=device
                    )
                    prof.step() # Mark the end of a step in profiling
            else:
                 # Call the main beamform function without profiling
                 bIQ3_vec_tensor = vectorized_beamform(
                    IQ3_tensor, xi_tensor, yi_tensor, zi_tensor, txdel3_tensor, element_pos_tensor,
                    fs, fc, num_elements, c=speed_of_sound, device=device
                 )


            # Existing timing code
            torch.cuda.synchronize()
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Beamforming time on CUDA (JIT, optimal chunking): {elapsed_time:.4f} seconds per volume")

            # Convert output tensor to numpy array after profiling if needed
            bIQ3_vec_np = bIQ3_vec_tensor.cpu().numpy()

        else: # CPU case
            # Call the main beamform function with tensors and default optimal chunk sizes
            start_time = time.time()
            bIQ3_vec_tensor = vectorized_beamform(
                 IQ3_tensor, xi_tensor, yi_tensor, zi_tensor, txdel3_tensor, element_pos_tensor,
                 fs, fc, num_elements, c=speed_of_sound, device=device
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Beamforming time on CPU (JIT, optimal chunking): {elapsed_time:.4f} seconds per volume")
            # Convert output tensor to numpy array
            bIQ3_vec_np = bIQ3_vec_tensor.cpu().numpy()


        print(f"Vectorized beamformed I/Q data generated. Shape: {bIQ3_vec_np.shape}")

        # Add visualization code here if needed, e.g., using matplotlib
        # import matplotlib.pyplot as plt
        # ... plot slices of bIQ3_vec_np ...

    except FileNotFoundError:
        print(f"Error: Data files not found in '{data_dir}'. Please run simulate_data.py first.")
    except ImportError:
         print("Error: Required libraries (numpy, torch, pymust) not found. Please install them.")
    except Exception as e:
        print(f"An error occurred during example usage: {e}")