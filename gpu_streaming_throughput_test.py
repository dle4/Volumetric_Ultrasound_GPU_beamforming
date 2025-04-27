import numpy as np
import os
import torch
import time
from vectorized_beamformer3D import vectorized_beamform # Import the updated beamformer

def test_gpu_quadrant_batch_throughput(data_dir="simulated_data"):
    """
    Tests the throughput of the GPU 3D beamformer using a single batch of 2 volumes,
    with internal XY chunking (quadrants) and Z-chunking (size 1).
    Includes accuracy checks against a reference.
    """
    print("\n--- Running GPU Quadrant Batch Throughput Test (Single Precision) ---")

    # --- Environment Setup & Verification ---
    print(f"Checking CUDA availability: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot perform GPU quadrant batch throughput test.")
        print("--- GPU Quadrant Batch Throughput Test Skipped ---")
        return

    device = 'cuda'
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Testing GPU quadrant batch throughput on device: {device} with a batch of 2 volumes.")

    try:
        # Load data
        print(f"Loading data from '{data_dir}'...")
        xi = np.load(os.path.join(data_dir, 'xi.npy'))
        yi = np.load(os.path.join(data_dir, 'yi.npy'))
        zi = np.load(os.path.join(data_dir, 'zi.npy'))
        txdel3 = np.load(os.path.join(data_dir, 'txdel3.npy'))
        param_data = np.load(os.path.join(data_dir, 'param_data.npy'), allow_pickle=True).item()
        RF3 = np.load(os.path.join(data_dir, 'RF3.npy'))
        bIQ3_ref = np.load(os.path.join(data_dir, 'bIQ3_ref.npy')) # Load reference for accuracy check

        print("Data loaded successfully.")

        # Demodulate RF data
        print("Demodulating RF data...")
        import pymust # Import pymust here
        IQ3_single_volume = pymust.rf2iq(RF3, param_data['fs'], param_data['fc'])
        print(f"Single volume I/Q data shape: {IQ3_single_volume.shape}")

        # --- Create a batch of volumes ---
        print("\nCreating a batch of volumes...")
        # Assuming IQ3_single_volume is (num_samples, num_elements)
        # We need to add a batch dimension and repeat the single volume twice
        batched_IQ = np.tile(IQ3_single_volume[np.newaxis, :, :], (50, 1, 1))
        print(f"Created batch data with shape: {batched_IQ.shape}")

        # --- Prepare for Beamforming ---
        num_samples, num_elements = IQ3_single_volume.shape
        grid_shape = xi.shape
        Nx, Ny, Nz = grid_shape
        output_volume_shape = (Nx, Ny, Nz) # Shape of a single beamformed volume

        # Convert inputs to GPU tensors once (single precision)
        batched_IQ_gpu = torch.from_numpy(batched_IQ.astype(np.complex64)).to(device)
        xi_gpu = torch.from_numpy(xi.astype(np.float32)).to(device)
        yi_gpu = torch.from_numpy(yi.astype(np.float32)).to(device)
        zi_gpu = torch.from_numpy(zi.astype(np.float32)).to(device)
        txdel_gpu = torch.from_numpy(txdel3.astype(np.float32)).to(device)

        # Convert element_pos to GPU tensor once (single precision)
        element_pos_gpu = torch.from_numpy(param_data['elements'].astype(np.float32)).to(device)

        # Create param_data dictionary with GPU tensor for elements
        param_data_gpu = {'fs': param_data['fs'], 'fc': param_data['fc'], 'elements': element_pos_gpu, 'Nelements': num_elements}

        # --- Run Beamformer with Timing ---
        print("\nRunning vectorized_beamform with a batch of N volumes and internal XY/Z chunking...")

        torch.cuda.synchronize()
        start_time = time.time()

        # Run beamformer on the GPU batch with internal XY and Z chunking
        # Set z_chunk_size to 1 and x_chunk_size/y_chunk_size for quadrants
        batched_bIQ3_vec_gpu = vectorized_beamform(
            batched_IQ_gpu,
            xi_gpu, yi_gpu, zi_gpu, txdel_gpu,
            param_data_gpu, # Pass the GPU version of param_data
            c=1540, # Use default value for c
            device=device,
            z_chunk_size=1, # Set z_chunk_size to 1
            x_chunk_size=Nx//2, # Set x_chunk_size for quadrants
            y_chunk_size=Ny//2, # Set y_chunk_size for quadrants
            input_on_gpu=True,
            output_on_gpu=True
        )

        torch.cuda.synchronize()
        end_time = time.time()
        elapsed_time_sec = end_time - start_time

        # Transfer output back to CPU for accuracy check
        batched_bIQ3_vec_cpu = batched_bIQ3_vec_gpu.cpu().numpy()
        
        print(f"\nBatch beamforming complete. Output shape: {batched_bIQ3_vec_cpu.shape}")
        print(f"Total execution time for batch of volumes: {elapsed_time_sec:.4f} seconds")

        # --- Accuracy Check (Comparing each volume of batch to reference) ---
        print("\nPerforming accuracy checks (comparing each volume of batch to reference)...")

        # Convert reference to complex64 for comparison
        bIQ3_ref_c64 = bIQ3_ref.astype(np.complex64)

        # Compare each volume of the batched output with the reference
        all_volumes_close = True
        for i in range(batched_bIQ3_vec_cpu.shape[0]):
            are_close = np.allclose(batched_bIQ3_vec_cpu[i], bIQ3_ref_c64, rtol=1e-4, atol=1e-5) # Adjust tolerance for single precision
            print(f"volume {i} of batch is close to reference (np.allclose): {are_close}")
            if not are_close:
                 all_volumes_close = False
                 max_diff = np.max(np.abs(batched_bIQ3_vec_cpu[i] - bIQ3_ref_c64))
                 print(f"Maximum absolute difference in volume {i}: {max_diff}")

        if all_volumes_close:
            print("All volumes in the batch are close to the reference.")
        else:
            print("At least one volume in the batch does not match the reference.")


        # --- Report Throughput ---
        num_volumes_in_batch = batched_IQ.shape[0]
        if elapsed_time_sec > 0:
            throughput_fps = num_volumes_in_batch / elapsed_time_sec
            print(f"\nThroughput (batch of {num_volumes_in_batch}): {throughput_fps:.2f} volumes per second (fps)")
        else:
            print("Execution time is zero, cannot calculate throughput.")


    except FileNotFoundError:
        print(f"Error: Data files not found in '{data_dir}'. Please run simulate_data.py first.")
    except ImportError:
         print("Error: Required libraries (numpy, torch, pymust) not found. Please install them.")
    except Exception as e:
        print(f"An error occurred during GPU quadrant batch throughput test: {e}")
        import traceback
        traceback.print_exc()

    print("--- GPU Quadrant Batch Throughput Test Complete ---")


if __name__ == '__main__':
    test_gpu_quadrant_batch_throughput()