import numpy as np
import os
import torch # Import PyTorch
import time # Import time module
import torch.profiler # Import torch.profiler

def vectorized_beamform(iq_data, xi, yi, zi, txdel, param_data, c=1540, device='cpu', z_chunk_size=4, x_chunk_size=-1, y_chunk_size=-1, input_on_gpu=False, output_on_gpu=False):
    """
    Performs vectorized Delay-and-Sum beamforming on I/Q data using PyTorch,
    matching the delay calculation and I/Q phase rotation from pymust.dasmtx3,
    with chunking along the Z-axis and optionally along X and Y axes to manage memory.
    Uses single precision (complex64/float32).

    Args:
        iq_data (np.ndarray or torch.Tensor): Input I/Q data. Shape: (batch_size, num_samples, num_elements) or (num_samples, num_elements)
                                             Can be NumPy array or PyTorch Tensor.
        xi (np.ndarray or torch.Tensor): X-coordinates of the beamforming grid. Shape: (Nx, Ny, Nz)
        yi (np.ndarray or torch.Tensor): Y-coordinates of the beamforming grid. Shape: (Nx, Ny, Nz)
        zi (np.ndarray or torch.Tensor): Z-coordinates of the beamforming grid. Shape: (Nx, Ny, Nz)
        txdel (np.ndarray or torch.Tensor): Transmit delays for each element. Shape: (1, num_elements) or (num_tx_events, num_elements)
                                          (Assuming single transmit event for now based on simulate_data.py)
        param_data (dict): Dictionary containing parameters like 'fs', 'fc', 'elements'.
                           'elements' shape: (2, num_elements) -> [x_coords, y_coords]
        c (float, optional): Speed of sound in m/s. Defaults to 1540.
        device (str, optional): The device to perform computation on ('cpu' or 'cuda'). Defaults to 'cpu'.
        z_chunk_size (int, optional): The number of Z-slices to process in each chunk. Defaults to 4.
        x_chunk_size (int, optional): The number of X-points to process in each chunk. Defaults to -1 (no X chunking).
        y_chunk_size (int, optional): The number of Y-points to process in each chunk. Defaults to -1 (no Y chunking).
        input_on_gpu (bool, optional): If True, assumes input tensors are already on the GPU. Skips HtoD transfer. Defaults to False.
        output_on_gpu (bool, optional): If True, returns the output tensor on the GPU. Skips DtoH transfer. Defaults to False.
    Returns:
        np.ndarray or torch.Tensor: Beamformed I/Q data. Shape: (batch_size, Nx, Ny, Nz) or (Nx, Ny, Nz)
                                    Returns NumPy array if output_on_gpu is False, otherwise returns PyTorch Tensor on the specified device.
    """
    print(f"Starting vectorized beamforming (PyTorch implementation) on device: {device} with single precision...")

    # --- Input Validation and Parameter Extraction ---
    fs = param_data['fs']
    fc = param_data['fc'] # Center frequency for I/Q phase rotation
    wc = 2 * np.pi * fc
    num_elements = param_data['Nelements']

    # Handle batch dimension and convert/move iq_data to PyTorch tensor on device (complex64)
    if not input_on_gpu:
        if isinstance(iq_data, np.ndarray):
            if iq_data.ndim == 3:
                batch_size, num_samples, num_elements_iq = iq_data.shape
                is_batch = True
                iq_tensor = torch.from_numpy(iq_data).to(torch.complex64).to(device)
            elif iq_data.ndim == 2:
                num_samples, num_elements_iq = iq_data.shape
                batch_size = 1
                iq_tensor = torch.from_numpy(iq_data).unsqueeze(0).to(torch.complex64).to(device) # Add batch dim
                is_batch = False
            else:
                raise ValueError("iq_data must have 2 or 3 dimensions (num_samples, num_elements) or (batch_size, num_samples, num_elements)")
        elif isinstance(iq_data, torch.Tensor):
             if iq_data.ndim == 3:
                batch_size, num_samples, num_elements_iq = iq_data.shape
                is_batch = True
                iq_tensor = iq_data.to(torch.complex64).to(device)
             elif iq_data.ndim == 2:
                num_samples, num_elements_iq = iq_data.shape
                batch_size = 1
                iq_tensor = iq_data.unsqueeze(0).to(torch.complex64).to(device) # Add batch dim
                is_batch = False
             else:
                raise ValueError("iq_data must have 2 or 3 dimensions (num_samples, num_elements) or (batch_size, num_samples, num_elements)")
        else:
            raise TypeError("iq_data must be a NumPy array or a PyTorch Tensor")
    else: # input_on_gpu is True, assume iq_tensor is already a PyTorch Tensor on the correct device
        if not isinstance(iq_data, torch.Tensor):
             raise TypeError("When input_on_gpu is True, iq_data must be a PyTorch Tensor")
        if iq_data.ndim == 3:
            batch_size, num_samples, num_elements_iq = iq_data.shape
            is_batch = True
            iq_tensor = iq_data.to(torch.complex64) # Ensure correct dtype
        elif iq_data.ndim == 2:
            num_samples, num_elements_iq = iq_data.shape
            batch_size = 1
            iq_tensor = iq_data.unsqueeze(0).to(torch.complex64) # Add batch dim and ensure correct dtype
        else:
            raise ValueError("iq_data must have 2 or 3 dimensions (num_samples, num_elements) or (batch_size, num_samples, num_elements)")


    if num_elements_iq != num_elements:
        raise ValueError(f"Mismatch between number of elements in iq_data ({num_elements_iq}) and param_data ({num_elements})")

    # Handle element_pos - can be NumPy array or PyTorch Tensor (float32)
    element_pos = param_data['elements']
    if isinstance(element_pos, np.ndarray):
        element_coords_3d = torch.zeros((1, num_elements, 3), dtype=torch.float32, device=device)
        element_coords_3d[0, :, 0] = torch.from_numpy(element_pos[0, :]).to(torch.float32).to(device) # x-coords
        element_coords_3d[0, :, 1] = torch.from_numpy(element_pos[1, :]).to(torch.float32).to(device) # y-coords
    elif isinstance(element_pos, torch.Tensor):
        # Assuming element_pos tensor is already on the correct device if input_on_gpu is True
        element_coords_3d = torch.zeros((1, num_elements, 3), dtype=torch.float32, device=device)
        element_coords_3d[0, :, 0] = element_pos[0, :].to(torch.float32) # x-coords
        element_coords_3d[0, :, 1] = element_pos[1, :].to(torch.float32) # y-coords
    else:
        raise TypeError("param_data['elements'] must be a NumPy array or a PyTorch Tensor")


    # Convert/move other inputs to PyTorch tensors and move to device if not already on GPU (float32)
    if not input_on_gpu:
        xi_tensor = torch.from_numpy(xi).to(torch.float32).to(device)
        yi_tensor = torch.from_numpy(yi).to(torch.float32).to(device)
        zi_tensor = torch.from_numpy(zi).to(torch.float32).to(device)
        txdel_tensor = torch.from_numpy(txdel).to(torch.float32).to(device)
    else: # input_on_gpu is True, assume other inputs are also PyTorch Tensors on the correct device
        if not (isinstance(xi, torch.Tensor) and isinstance(yi, torch.Tensor) and isinstance(zi, torch.Tensor) and isinstance(txdel, torch.Tensor)):
             raise TypeError("When input_on_gpu is True, xi, yi, zi, and txdel must also be PyTorch Tensors")
        xi_tensor = xi.to(torch.float32).to(device)
        yi_tensor = yi.to(torch.float32).to(device)
        zi_tensor = zi.to(torch.float32).to(device)
        txdel_tensor = txdel.to(torch.float32).to(device)


    if txdel_tensor.shape[1] != num_elements:
         raise ValueError(f"Mismatch between number of elements in txdel ({txdel_tensor.shape[1]}) and param_data ({num_elements})")
    # Assuming single transmit event for now, matching simulate_data.py
    if txdel_tensor.shape[0] != 1:
        print("Warning: txdel has more than one row, assuming single transmit event and using the first row.")
        txdel_tensor = txdel_tensor[0:1, :] # Use only the first row

    grid_shape = xi_tensor.shape # (Nx, Ny, Nz)
    Nx, Ny, Nz = grid_shape

    # Determine chunking for X and Y dimensions
    if x_chunk_size == -1 or x_chunk_size > Nx:
        x_chunk_size = Nx
    if y_chunk_size == -1 or y_chunk_size > Ny:
        y_chunk_size = Ny

    x_starts = range(0, Nx, x_chunk_size)
    x_ends = [min(x_start + x_chunk_size, Nx) for x_start in x_starts]
    y_starts = range(0, Ny, y_chunk_size)
    y_ends = [min(y_start + y_chunk_size, Ny) for y_start in y_starts]


    # Initialize the output tensor on the specified device (complex64)
    beamformed_iq = torch.zeros((batch_size, Nx, Ny, Nz), dtype=torch.complex64, device=device)

    # --- Chunking along Z-axis ---
    # Disable internal Z-chunking if input is on GPU and z_chunk_size is effectively disabling chunking
    if input_on_gpu and z_chunk_size >= Nz: # Treat z_chunk_size >= Nz as no internal chunking
        print("Input is on GPU and z_chunk_size is effectively disabled. Processing entire Z-axis.")
        z_starts = [0]
        z_ends = [Nz]
        current_Nzs = [Nz]
    else:
        print(f"Processing grid in chunks of size {z_chunk_size} along Z-axis...")
        z_starts = range(0, Nz, z_chunk_size)
        z_ends = [min(z_start + z_chunk_size, Nz) for z_start in z_starts]
        current_Nzs = [z_end - z_start for z_start, z_end in zip(z_starts, z_ends)]

    # --- Nested Chunking along X and Y axes ---
    print(f"Processing grid in chunks of size ({x_chunk_size}, {y_chunk_size}) along X and Y axes within each Z chunk...")

    for z_start, z_end, current_Nz in zip(z_starts, z_ends, current_Nzs):
        if current_Nz == 0:
            continue

        for x_start, x_end in zip(x_starts, x_ends):
            current_Nx = x_end - x_start
            if current_Nx == 0:
                continue

            for y_start, y_end in zip(y_starts, y_ends):
                current_Ny = y_end - y_start
                if current_Ny == 0:
                    continue

                num_chunk_points = current_Nx * current_Ny * current_Nz

                # Select grid points for the current XY-Z chunk
                xi_chunk = xi_tensor[x_start:x_end, y_start:y_end, z_start:z_end]
                yi_chunk = yi_tensor[x_start:x_end, y_start:y_end, z_start:z_end]
                zi_chunk = zi_tensor[x_start:x_end, y_start:y_end, z_start:z_end]

                # Reshape chunk grid points for broadcasting: (current_Nx*current_Ny*current_Nz, 1, 3)
                grid_points_chunk = torch.stack([xi_chunk.ravel(), yi_chunk.ravel(), zi_chunk.ravel()], dim=-1).unsqueeze(1)

                # --- Delay Calculation (Vectorized with PyTorch) - Matching dasmtx3 logic ---
                # print(f"Calculating distances and delays for chunk Z={z_start}:{z_end}, X={x_start}:{x_end}, Y={y_start}:{y_end}...")

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
                # print(f"Performing linear interpolation and apodization for chunk Z={z_start}:{z_end}, X={x_start}:{x_end}, Y={y_start}:{y_end}...")

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
                iq_tensor_transposed = iq_tensor.transpose(1, 2)

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

                # --- I/Q Phase Rotation (Matching dasmtx3 logic) ---
                # Apply phase rotation: exp(1j * wc * total_delay)
                # total_delay_chunk shape: (current_Nx*current_Ny*current_Nz, num_elements)
                # Reshape total_delay_chunk for broadcasting: (1, num_elements, current_Nx*current_Ny*current_Nz)
                total_delay_reshaped_chunk = total_delay_chunk.unsqueeze(0).transpose(1, 2)
                phase_rotation_chunk = torch.exp(1j * wc * total_delay_reshaped_chunk) # Shape: (1, num_elements, current_Nx*current_Ny*current_Nz)

                # Apply phase rotation to apodized samples
                rotated_samples_chunk = apodized_samples_chunk * phase_rotation_chunk # Shape: (batch_size, num_elements, current_Nx*current_Ny*current_Nz)

                # --- Summation ---
                # print(f"Performing summation for chunk Z={z_start}:{z_end}, X={x_start}:{x_end}, Y={y_start}:{y_end}...")
                beamformed_iq_flat_chunk = torch.sum(rotated_samples_chunk, dim=1) # Shape: (batch_size, current_Nx*current_Ny*current_Nz)

                # Reshape to match the chunk grid dimensions
                beamformed_iq_chunk = beamformed_iq_flat_chunk.reshape((batch_size, current_Nx, current_Ny, current_Nz)) # Shape: (batch_size, current_Nx, current_Ny, current_Nz)

                # Store the beamformed chunk in the final output tensor
                beamformed_iq[:, x_start:x_end, y_start:y_end, z_start:z_end] = beamformed_iq_chunk


    # --- Return the final beamformed volume ---
    if not output_on_gpu:
        return beamformed_iq.cpu().numpy() # Move to CPU and convert to NumPy array
    else:
        return beamformed_iq # Return PyTorch Tensor on the specified device


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
        # Load data
        print(f"Loading data from '{data_dir}'...")
        xi = np.load(os.path.join(data_dir, 'xi.npy'))
        yi = np.load(os.path.join(data_dir, 'yi.npy'))
        zi = np.load(os.path.join(data_dir, 'zi.npy'))
        txdel3 = np.load(os.path.join(data_dir, 'txdel3.npy'))
        param_data = np.load(os.path.join(data_dir, 'param_data.npy'), allow_pickle=True).item()
        bIQ3_ref = np.load(os.path.join(data_dir, 'bIQ3_ref.npy'))
        print("Data loaded successfully.")

        # Demodulate RF data (as per plan, this is done outside the beamformer)
        print("Demodulating RF data...")
        import pymust # Import pymust here for the example usage
        # Need to load RF data to demodulate
        RF3 = np.load(os.path.join(data_dir, 'RF3.npy'))
        IQ3 = pymust.rf2iq(RF3, param_data['fs'], param_data['fc'])
        print(f"I/Q data generated. Shape: {IQ3.shape}")

        # Run vectorized beamformer with chunking
        print("Running vectorized_beamform with Z-chunking...")
        # You can adjust z_chunk_size here to experiment with different chunk sizes

        if device == 'cuda':
            # Existing timing code
            torch.cuda.synchronize()
            start_time = time.time()

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
                bIQ3_vec = vectorized_beamform(IQ3, xi, yi, zi, txdel3, param_data, device=device, z_chunk_size=1) # Example chunk size
                prof.step() # Mark the end of a step in profiling

            # Existing timing code
            torch.cuda.synchronize()
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Beamforming time on CUDA: {elapsed_time:.4f} seconds per volume")
        else:
            # If not CUDA, just run the beamformer without profiling
            bIQ3_vec = vectorized_beamform(IQ3, xi, yi, zi, txdel3, param_data, device=device, z_chunk_size=1) # Example chunk size


        print(f"Vectorized beamformed I/Q data generated. Shape: {bIQ3_vec.shape}")

        # --- Comparison (Step 3) ---
        print("\nComparing vectorized output with PyMUST reference (using complex64)...")

        # Convert reference to complex64 for comparison
        bIQ3_ref_c64 = bIQ3_ref.astype(np.complex64)

        # Option 1: Full volume comparison (can be slow)
        try:
            print("Comparing full volumes using np.allclose...")
            are_close = np.allclose(bIQ3_vec, bIQ3_ref_c64, rtol=1e-4, atol=1e-5) # Adjust tolerance for single precision
            print(f"Volumes are close (np.allclose): {are_close}")
            if not are_close:
                 max_diff = np.max(np.abs(bIQ3_vec - bIQ3_ref_c64))
                 print(f"Maximum absolute difference: {max_diff}")
        except MemoryError:
            print("MemoryError during full volume comparison. Skipping.")
            are_close = False # Assume not close if comparison failed
        except Exception as e:
             print(f"An error occurred during example usage: {e}")
             are_close = False


        # Option 2: Center slice comparison (faster)
        if not are_close: # Only run if full comparison failed or returned False
            print("\nComparing center XZ slices...")
            nx, ny, nz = xi.shape
            center_y_idx = ny // 2
            slice_vec = bIQ3_vec[:, center_y_idx, :]
            slice_ref = bIQ3_ref_c64[:, center_y_idx, :]
            slice_close = np.allclose(slice_vec, slice_ref, rtol=1e-4, atol=1e-5) # Adjust tolerance for single precision
            print(f"Center XZ slices are close (np.allclose): {slice_close}")
            if not slice_close:
                 max_diff_slice = np.max(np.abs(slice_vec - slice_ref))
                 print(f"Maximum absolute difference in slice: {max_diff_slice}")

        # --- Visualization (Optional) ---
        # Add visualization code here if needed, e.g., using matplotlib
        # import matplotlib.pyplot as plt
        # ... plot slices of bIQ3_vec and bIQ3_ref ...

    except FileNotFoundError:
        print(f"Error: Data files not found in '{data_dir}'. Please run simulate_data.py first.")
    except ImportError:
         print("Error: Required libraries (numpy, torch, pymust) not found. Please install them.")
    except Exception as e:
        print(f"An error occurred during example usage: {e}")