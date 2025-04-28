import numpy as np
import os
import torch
import time
from vectorized_beamformer3D import vectorized_beamform # Import the updated beamformer

def test_gpu_quadrant_batch_throughput(data_dir="simulated_data"):
    """
    Tests the throughput of the GPU 3D beamformer using a single batch of volumes,
    with internal XY chunking (quadrants) and Z-chunking (size 1).
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
    print(f"Testing GPU quadrant batch throughput on device: {device} with a batch of volumes.")

    try:
        # Load data
        print(f"Loading data from '{data_dir}'...")
        xi_np = np.load(os.path.join(data_dir, 'xi.npy'))
        yi_np = np.load(os.path.join(data_dir, 'yi.npy'))
        zi_np = np.load(os.path.join(data_dir, 'zi.npy'))
        txdel3_np = np.load(os.path.join(data_dir, 'txdel3.npy'))
        param_data = np.load(os.path.join(data_dir, 'param_data.npy'), allow_pickle=True).item()
        

        print("Data loaded successfully.")

        # Demodulate RF data
        print("Demodulating RF data...")
        import pymust # Import pymust here
        RF3_np = np.load(os.path.join(data_dir, 'RF3.npy'))
        IQ3_np = pymust.rf2iq(RF3_np, param_data['fs'], param_data['fc'])
        print(f"Single volume I/Q data shape: {IQ3_np.shape}")

        # --- Create a batch of volumes ---
        print("\nCreating a batch of volumes...")
        # Assuming IQ3_np is (num_samples, num_elements)
        # We need to add a batch dimension and repeat the single volume
        batched_IQ_np = np.tile(IQ3_np[np.newaxis, :, :], (50, 1, 1)) # Example batch size of 50
        print(f"Created batch data with shape: {batched_IQ_np.shape}")

        # --- Prepare for Beamforming ---
        num_samples, num_elements = IQ3_np.shape
        grid_shape = xi_np.shape
        Nx, Ny, Nz = grid_shape
        output_volume_shape = (Nx, Ny, Nz) # Shape of a single beamformed volume

        # Extract parameters and convert all inputs to tensors and move to device
        fs = param_data['fs']
        fc = param_data['fc']
        num_elements_param = param_data['Nelements'] # Use num_elements from param_data
        element_pos_np = param_data['elements']
        speed_of_sound = 1540.0 # Use default value for c

        # Ensure num_elements match
        if num_elements != num_elements_param:
             raise ValueError(f"Mismatch between num_elements from IQ data ({num_elements}) and param_data ({num_elements_param})")


        batched_IQ_tensor = torch.from_numpy(batched_IQ_np.astype(np.complex64)).to(device)
        xi_tensor = torch.from_numpy(xi_np.astype(np.float32)).to(device)
        yi_tensor = torch.from_numpy(yi_np.astype(np.float32)).to(device)
        zi_tensor = torch.from_numpy(zi_np.astype(np.float32)).to(device)
        txdel_tensor = torch.from_numpy(txdel3_np.astype(np.float32)).to(device)
        element_pos_tensor = torch.from_numpy(element_pos_np.astype(np.float32)).to(device)

        # Determine batch size from the batched IQ tensor
        batch_size = batched_IQ_tensor.shape[0]


        # --- Run Beamformer with Timing ---
        print("\nRunning vectorized_beamform with a batch of volumes and internal XY/Z chunking...")

        torch.cuda.synchronize()
        start_time = time.time()

        # Run beamformer on the GPU batch with internal XY and Z chunking
        # Use specific chunk sizes for this test (quadrants and z=1)
        batched_bIQ3_vec_tensor = vectorized_beamform(
            batched_IQ_tensor,
            xi_tensor, yi_tensor, zi_tensor, txdel_tensor, element_pos_tensor,
            fs, fc, num_elements, c=speed_of_sound, device=device,
            z_chunk_size=8, # Set z_chunk_size to 1 for this test
            x_chunk_size=8, # Set x_chunk_size for quadrants
            y_chunk_size=16  # Set y_chunk_size for quadrants
        )

        torch.cuda.synchronize()
        end_time = time.time()
        elapsed_time_sec = end_time - start_time

        # Transfer output back to CPU for accuracy check
        batched_bIQ3_vec_cpu = batched_bIQ3_vec_tensor.cpu().numpy()

        print(f"\nBatch beamforming complete. Output shape: {batched_bIQ3_vec_cpu.shape}")
        print(f"Total execution time for batch of {batch_size} volumes: {elapsed_time_sec:.4f} seconds")

    

        # --- Report Throughput ---
        if elapsed_time_sec > 0:
            throughput_fps = batch_size / elapsed_time_sec
            print(f"\nThroughput (batch of {batch_size}): {throughput_fps:.2f} volumes per second (fps)")
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