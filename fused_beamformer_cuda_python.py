import numpy as np
import os
import time
import cuda # Import cuda-python
import cuda.nvrtc # Import NVRTC for runtime compilation
import cuda.cudadrv # Import CUDA driver API

# Load the CUDA kernel source code
def load_kernel_source(filename):
    with open(filename, 'r') as f:
        return f.read()

# Compile the CUDA kernel
def compile_kernel(source_code, kernel_name):
    # Create a program
    prog = cuda.nvrtc.createProgram(source_code, kernel_name)

    # Compile the program
    # Options can be added here, e.g., for specifying CUDA architecture
    try:
        cuda.nvrtc.compileProgram(prog, [])
    except Exception as e:
        # Get compilation log to see errors
        log = cuda.nvrtc.getProgramLog(prog)
        print("CUDA Kernel Compilation Failed:")
        print(log.decode())
        raise e

    # Get PTX from the program
    ptx = cuda.nvrtc.getPTX(prog)
    return ptx

# Main beamforming function using cuda-python
def fused_vectorized_beamform_cuda_python(
    iq_data_np: np.ndarray,         # Input I/Q data (NumPy array, complex64) - Shape: (num_samples, num_elements)
    xi_np: np.ndarray,              # X-coordinates of the beamforming grid (NumPy array, float32) - Shape: (Nx, Ny, Nz)
    yi_np: np.ndarray,              # Y-coordinates of the beamforming grid (NumPy array, float32) - Shape: (Nx, Ny, Nz)
    zi_np: np.ndarray,              # Z-coordinates of the beamforming grid (NumPy array, float32) - Shape: (Nx, Ny, Nz)
    txdel_np: np.ndarray,           # Transmit delays (NumPy array, float32) - Shape: (1, num_elements)
    element_pos_np: np.ndarray,     # Transducer element positions (NumPy array, float32) - Shape: (2, num_elements) - Assuming 2D for now, will need 3D
    fs: float,                      # Sampling frequency in Hz
    fc: float,                      # Center frequency for I/Q phase rotation in Hz
    num_elements: int,              # Number of transducer elements
    c: float = 1540.0,              # Speed of sound in m/s
    z_chunk_size: int = 8,
    x_chunk_size: int = 8,
    y_chunk_size: int = 16
):
    print("Starting fused vectorized beamforming (cuda-python implementation)...")

    # --- Initialize CUDA Context ---
    # Initialization is now handled in the example usage block

    # --- Load and Compile Kernel ---
    kernel_filename = "GPU_volumetric_beamforming/fused_beamform_kernel.cu"
    kernel_name = "fused_beamform_kernel"
    print(f"Loading and compiling CUDA kernel from '{kernel_filename}'...")
    kernel_source = load_kernel_source(kernel_filename)
    ptx = compile_kernel(kernel_source, kernel_name)
    print("Kernel compiled successfully.")

    # --- Load Module and Get Kernel Function ---
    module = cuda.cudadrv.moduleLoadData(ptx)
    kernel_func = cuda.cudadrv.moduleGetFunction(module, kernel_name)
    print("CUDA module loaded and kernel function obtained.")

    # --- Prepare Input Data (NumPy to CUDA) ---
    # Ensure data types are correct (float32 for floats, complex64 for complex)
    # Transpose IQ data to [elements][samples] layout for the kernel
    iq_data_transposed_np = iq_data_np.transpose().astype(np.complex62) # Use complex62 for float2 compatibility
    # Need to handle element_pos_np to be 3D (add a Z dimension, likely zeros)
    if element_pos_np.shape[0] == 2: # Assuming 2D [x, y]
        element_coords_3d_np = np.zeros((3, num_elements), dtype=np.float32)
        element_coords_3d_np[0:2, :] = element_pos_np.astype(np.float32)
    elif element_pos_np.shape[0] == 3: # Already 3D [x, y, z]
         element_coords_3d_np = element_pos_np.astype(np.float32)
    else:
        raise ValueError("element_pos_np must have shape (2, num_elements) or (3, num_elements)")

    # Flatten grid coordinates for chunk processing
    # These will be chunked later, but allocate space for the full grid for now if needed,
    # or handle allocation within the chunking loop. Let's handle within the loop.

    # --- Extract Parameters ---
    num_samples = iq_data_np.shape[0]
    # num_elements = iq_data_np.shape[1] # Already provided

    grid_shape = xi_np.shape # (Nx, Ny, Nz)
    Nx, Ny, Nz = grid_shape

    # Determine chunking for X and Y dimensions
    if x_chunk_size == -1 or x_chunk_size > Nx:
        x_chunk_size = Nx
    if y_chunk_size == -1 or y_chunk_size > Ny:
        y_chunk_size = Ny
    z_chunk_size = min(z_chunk_size, Nz)

    # Initialize output tensor on host (NumPy array)
    beamformed_iq_np = np.zeros((Nx, Ny, Nz), dtype=np.complex64) # Output is (Nx, Ny, Nz)

    # --- Chunking along Z, X, Y axes ---
    print(f"Processing grid in chunks of size ({x_chunk_size}, {y_chunk_size}, {z_chunk_size}) along X, Y, Z axes...")

    z_start = 0
    while z_start < Nz:
        z_end = min(z_start + z_chunk_size, Nz)
        current_Nz = z_end - z_start
        if current_Nz == 0: break

        x_start = 0
        while x_start < Nx:
            x_end = min(x_start + x_chunk_size, Nx)
            current_Nx = x_end - x_start
            if current_Nx == 0: break

            y_start = 0
            while y_start < Ny:
                y_end = min(y_start + y_chunk_size, Ny)
                current_Ny = y_end - y_start
                if current_Ny == 0: break

                # Select grid points for the current XY-Z chunk (NumPy)
                xi_chunk_np = xi_np[x_start:x_end, y_start:y_end, z_start:z_end].ravel().astype(np.float32)
                yi_chunk_np = yi_np[x_start:x_end, y_start:y_end, z_start:z_end].ravel().astype(np.float32)
                zi_chunk_np = zi_np[x_start:x_end, y_start:y_end, z_start:z_end].ravel().astype(np.float32)
                num_chunk_points = current_Nx * current_Ny * current_Nz

                # --- Allocate GPU Memory for Chunk Inputs ---
                # Allocate memory for the full transposed IQ data, element positions, and tx delays once
                # and reuse for each chunk.
                if 'gpu_iq_data' not in locals():
                     gpu_iq_data = cuda.cudadrv.mem_alloc(iq_data_transposed_np.nbytes)
                     cuda.cudadrv.memcpy_htod(gpu_iq_data, iq_data_transposed_np)

                     gpu_element_x = cuda.cudadrv.mem_alloc(element_coords_3d_np[0, :].nbytes)
                     cuda.cudadrv.memcpy_htod(gpu_element_x, element_coords_3d_np[0, :])
                     gpu_element_y = cuda.cudadrv.mem_alloc(element_coords_3d_np[1, :].nbytes)
                     cuda.cudadrv.memcpy_htod(gpu_element_y, element_coords_3d_np[1, :])
                     gpu_element_z = cuda.cudadrv.mem_alloc(element_coords_3d_np[2, :].nbytes)
                     cuda.cudadrv.memcpy_htod(gpu_element_z, element_coords_3d_np[2, :])

                     gpu_tx_delays = cuda.cudadrv.mem_alloc(txdel_np[0, :].nbytes) # Assuming single tx event
                     cuda.cudadrv.memcpy_htod(gpu_tx_delays, txdel_np[0, :].astype(np.float32))


                # Allocate memory for chunk-specific grid points and output
                gpu_grid_x = cuda.cudadrv.mem_alloc(xi_chunk_np.nbytes)
                gpu_grid_y = cuda.cudadrv.mem_alloc(yi_chunk_np.nbytes)
                gpu_grid_z = cuda.cudadrv.mem_alloc(zi_chunk_np.nbytes)
                gpu_beamformed_iq_chunk = cuda.cudadrv.mem_alloc(num_chunk_points * np.dtype(np.complex64).itemsize) # Output is complex64

                # --- Copy Chunk Input Data Host -> Device ---
                cuda.cudadrv.memcpy_htod(gpu_grid_x, xi_chunk_np)
                cuda.cudadrv.memcpy_htod(gpu_grid_y, yi_chunk_np)
                cuda.cudadrv.memcpy_htod(gpu_grid_z, zi_chunk_np)

                # --- Define Grid and Block Dimensions ---
                # Simple 1D grid, each thread processes one point in the chunk
                threads_per_block = 256 # Example block size
                blocks_per_grid = (num_chunk_points + threads_per_block - 1) // threads_per_block

                # --- Launch Kernel ---
                # Prepare kernel arguments (pointers and scalars)
                # Order must match the kernel function signature!
                kernel_args = [
                    gpu_iq_data,
                    gpu_grid_x,
                    gpu_grid_y,
                    gpu_grid_z,
                    gpu_element_x,
                    gpu_element_y,
                    gpu_element_z,
                    gpu_tx_delays,
                    gpu_beamformed_iq_chunk,
                    np.float32(fs),
                    np.float32(fc),
                    np.float32(c),
                    np.int32(num_samples),
                    np.int32(num_elements),
                    np.int32(num_chunk_points)
                ]

                # Launch the kernel
                # kernel_func(gridDim, blockDim, sharedMemSize, stream, args)
                kernel_func(
                    (blocks_per_grid, 1, 1), # Grid dimension
                    (threads_per_block, 1, 1), # Block dimension
                    0, # Shared memory size in bytes
                    None, # Stream (None for default stream)
                    kernel_args # Kernel arguments
                )

                # --- Copy Beamformed Chunk Data Device -> Host ---
                beamformed_iq_chunk_np = np.empty(num_chunk_points, dtype=np.complex64)
                cuda.cudadrv.memcpy_dtoh(beamformed_iq_chunk_np, gpu_beamformed_iq_chunk)

                # --- Free GPU Memory for Chunk Inputs/Output ---
                gpu_grid_x.free()
                gpu_grid_y.free()
                gpu_grid_z.free()
                gpu_beamformed_iq_chunk.free()

                # --- Reshape and Store the Beamformed Chunk ---
                # Reshape the flat chunk data back to its original chunk dimensions
                beamformed_iq_chunk_reshaped = beamformed_iq_chunk_np.reshape((current_Nx, current_Ny, current_Nz))

                # Store the chunk in the final output NumPy array
                beamformed_iq_np[x_start:x_end, y_start:y_end, z_start:z_end] = beamformed_iq_chunk_reshaped

                y_start += y_chunk_size
            x_start += x_chunk_size
        z_start += z_chunk_size

    # --- Free GPU Memory for Reused Buffers ---
    if 'gpu_iq_data' in locals():
        gpu_iq_data.free()
        gpu_element_x.free()
        gpu_element_y.free()
        gpu_element_z.free()
        gpu_tx_delays.free()

    # --- Pop CUDA Context (if explicitly pushed) ---
    # context.pop() # Pop the context if it was explicitly pushed

    print("Fused beamforming (cuda-python) complete.")
    return beamformed_iq_np # Return NumPy array

# Example Usage (requires saved data from simulate_data.py)
if __name__ == '__main__':
    print("\n--- Running Example Usage (cuda-python) ---")
    # --- Environment Setup & Verification ---
    print("\n--- Environment Setup & Verification ---")
    try:
        # Attempt to initialize the CUDA driver API
        cuda.cudadrv.cuInit(0)
        device = cuda.cudadrv.Device(0)
        print(f"CUDA is available. Using GPU: {device.name()}")
        # context = device.retain_primary_context()
        # context.push()
    except Exception as e:
        print(f"CUDA initialization failed: {e}")
        print("Cannot run CUDA example usage.")
        exit()
    print("Selected device: CUDA")
    # --- End Environment Setup & Verification ---

    data_dir = "simulated_data"
    try:
        # Load data (as NumPy arrays)
        print(f"Loading data from '{data_dir}'...")
        xi_np = np.load(os.path.join(data_dir, 'xi.npy'))
        yi_np = np.load(os.path.join(data_dir, 'yi.npy'))
        zi_np = np.load(os.path.join(data_dir, 'zi.npy'))
        txdel_np = np.load(os.path.join(data_dir, 'txdel3.npy')) # Assuming txdel3.npy is the correct one
        param_data = np.load(os.path.join(data_dir, 'param_data.npy'), allow_pickle=True).item()
        # Load IQ data (generated by simulate_data.py)
        # Note: simulate_data.py saves RF3.npy, need to run rf2iq if IQ3.npy is not saved
        # Assuming IQ3.npy exists or we regenerate it here for the example
        try:
             IQ_np = np.load(os.path.join(data_dir, 'IQ3.npy')) # Try loading saved IQ
             print("Loaded saved IQ data.")
        except FileNotFoundError:
             print("IQ3.npy not found, regenerating from RF3.npy...")
             RF_np = np.load(os.path.join(data_dir, 'RF3.npy'))
             import pymust # Import pymust here for the example usage
             IQ_np = pymust.rf2iq(RF_np, param_data['fs'], param_data['fc'])
             print(f"I/Q data regenerated. Shape: {IQ_np.shape}")


        print("Data loaded successfully.")

        # Extract parameters
        fs = param_data['fs']
        fc = param_data['fc']
        num_elements = param_data['Nelements']
        element_pos_np = param_data['elements'] # This is 2D [x, y] from simulate_data.py
        speed_of_sound = 1540.0 # Use default c

        # Run fused beamformer with cuda-python
        print("\n--- Running fused_vectorized_beamform_cuda_python ---")

        start_time = time.time()
        bIQ_fused_np = fused_vectorized_beamform_cuda_python(
            IQ_np, xi_np, yi_np, zi_np, txdel_np, element_pos_np,
            fs, fc, num_elements, c=speed_of_sound
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Fused beamforming time (cuda-python): {elapsed_time:.4f} seconds")

        print(f"Fused beamformed I/Q data generated. Shape: {bIQ_fused_np.shape}")

        # --- Verification (Optional) ---
        # Load reference beamformed data from simulate_data.py
        try:
            bIQ_ref_np = np.load(os.path.join(data_dir, 'bIQ3_ref.npy'))
            print(f"Loaded reference beamformed data. Shape: {bIQ_ref_np.shape}")

            # Compare outputs
            # Need to handle potential differences in shape if batch size was involved in PyTorch
            # Assuming the reference is (Nx, Ny, Nz) and our output is (Nx, Ny, Nz)
            if bIQ_fused_np.shape == bIQ_ref_np.shape:
                 # Use a tolerance for floating point comparisons
                 if np.allclose(bIQ_fused_np, bIQ_ref_np, rtol=1e-5, atol=1e-8):
                     print("Verification successful: Fused output matches reference output.")
                 else:
                     print("Verification failed: Fused output DOES NOT match reference output.")
                     # Optional: print max absolute difference
                     abs_diff = np.abs(bIQ_fused_np - bIQ_ref_np)
                     print(f"Max absolute difference: {np.max(abs_diff)}")
            else:
                 print(f"Verification skipped: Output shape {bIQ_fused_np.shape} does not match reference shape {bIQ_ref_np.shape}.")

        except FileNotFoundError:
            print("Reference beamformed data (bIQ3_ref.npy) not found. Skipping verification.")
        except Exception as e:
            print(f"An error occurred during verification: {e}")


    except FileNotFoundError:
        print(f"Error: Data files not found in '{data_dir}'. Please run simulate_data.py first.")
    except ImportError:
         print("Error: Required libraries (numpy, cuda, pymust) not found. Please install them.")
    except Exception as e:
        print(f"An error occurred during example usage: {e}")
        import traceback
        traceback.print_exc()

    # --- Pop CUDA Context (if explicitly pushed) ---
    # try:
    #     context.pop()
    # except Exception:
    #     pass # Ignore if context was not pushed or already popped