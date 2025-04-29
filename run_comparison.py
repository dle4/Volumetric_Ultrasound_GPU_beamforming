import numpy as np
import os
import torch
import time
import matplotlib.pyplot as plt
import pymust # Assuming pymust is installed and available

# Import the two beamformer implementations
from vectorized_beamformer3D import vectorized_beamform as original_vectorized_beamform
from vectorized_beamformer3D_ext import vectorized_beamform_ext as fused_vectorized_beamform

def run_comparison():
    """
    Runs visual and throughput comparisons between the original and fused
    vectorized 3D beamformers.
    """
    print("\n--- Starting Beamformer Comparison ---")

    # --- Environment Setup & Verification ---
    print("\n--- Environment Setup & Verification ---")
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("CUDA is not available. Cannot run CUDA comparison.")
        return # Exit if CUDA is not available
    print(f"Selected device: {device}")
    # --- End Environment Setup & Verification ---

    data_dir = "simulated_data"
    output_dir = "example_outputs"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # --- Load Data ---
        print(f"\n--- Loading data from '{data_dir}' ---")
        xi_np = np.load(os.path.join(data_dir, 'xi.npy'))
        yi_np = np.load(os.path.join(data_dir, 'yi.npy'))
        zi_np = np.load(os.path.join(data_dir, 'zi.npy'))
        txdel3_np = np.load(os.path.join(data_dir, 'txdel3.npy'))
        param_data = np.load(os.path.join(data_dir, 'param_data.npy'), allow_pickle=True).item()
        RF3_np = np.load(os.path.join(data_dir, 'RF3.npy'))
        print("Data loaded successfully.")

        # --- Extract Parameters ---
        fs = param_data['fs']
        fc = param_data['fc']
        num_elements = param_data['Nelements']
        element_pos_np = param_data['elements'] # Element positions for both beamformers
        speed_of_sound = 1540.0 # Use default c

        # --- Prepare Single IQ Volume for Visual Comparison ---
        print("\n--- Preparing single IQ volume ---")
        IQ3_np = pymust.rf2iq(RF3_np, fs, fc)
        print(f"Single I/Q volume generated. Shape: {IQ3_np.shape}")

        # Convert NumPy arrays to PyTorch tensors and move to device
        # Add batch dimension of 1 for consistency
        IQ3_tensor = torch.from_numpy(IQ3_np).to(torch.complex64).unsqueeze(0).to(device) # Shape: (1, num_samples, num_elements)
        xi_tensor = torch.from_numpy(xi_np).to(torch.float32).to(device)
        yi_tensor = torch.from_numpy(yi_np).to(torch.float32).to(device)
        zi_tensor = torch.from_numpy(zi_np).to(torch.float32).to(device)
        txdel3_tensor = torch.from_numpy(txdel3_np).to(torch.float32).to(device) # Shape: (1, num_elements)
        element_pos_tensor = torch.from_numpy(element_pos_np).to(torch.float32).to(device) # Shape: (2, num_elements)

        # Determine batch size and num_samples from the IQ tensor
        batch_size_single, num_samples, _ = IQ3_tensor.shape
        grid_shape = xi_tensor.shape
        Nx, Ny, Nz = grid_shape

        # --- Visual Comparison (Single Volume) ---
        print("\n--- Running Visual Comparison ---")

        # Run original vectorized beamformer (PyTorch JIT)
        print("Running original vectorized_beamform (PyTorch JIT)...")
        with torch.no_grad(): # No need to track gradients for comparison
            bIQ3_original_tensor = original_vectorized_beamform(
                 IQ3_tensor, xi_tensor, yi_tensor, zi_tensor, txdel3_tensor, element_pos_tensor,
                 fs, fc, num_elements, c=speed_of_sound, device=device
            )
        print("Original beamforming complete.")

        # Run fused vectorized beamformer (PyTorch C++ Extension)
        print("Running fused vectorized_beamform_ext (PyTorch C++ Extension)...")
        with torch.no_grad(): # No need to track gradients for comparison
             bIQ3_fused_tensor = fused_vectorized_beamform(
                IQ3_tensor, xi_tensor, yi_tensor, zi_tensor, txdel3_tensor, element_pos_tensor,
                fs, fc, num_elements, c=speed_of_sound, device=device
             )
        print("Fused extension beamforming complete.")

        # Convert output tensors to numpy arrays for plotting (remove batch dim)
        bIQ3_original_np = bIQ3_original_tensor.squeeze(0).abs().cpu().numpy()
        bIQ3_fused_np = bIQ3_fused_tensor.squeeze(0).abs().cpu().numpy()

        # Calculate center indices for slicing
        center_x_idx = Nx // 2
        center_y_idx = Ny // 2
        center_z_idx = Nz // 2

        # Plotting function for slices
        def plot_slice_comparison(slice_original, slice_fused, title, filename, extent, xlabel, ylabel):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            im1 = axes[0].imshow(slice_original.T, aspect='auto', cmap='gray', extent=extent)
            axes[0].set_title('Original PyTorch JIT')
            axes[0].set_xlabel(xlabel)
            axes[0].set_ylabel(ylabel)
            fig.colorbar(im1, ax=axes[0])

            im2 = axes[1].imshow(slice_fused.T, aspect='auto', cmap='gray', extent=extent)
            axes[1].set_title('Fused C++ Extension')
            axes[1].set_xlabel(xlabel)
            axes[1].set_ylabel(ylabel)
            fig.colorbar(im2, ax=axes[1])

            fig.suptitle(title)
            plt.savefig(os.path.join(output_dir, filename))
            print(f"Saved {filename}")
            plt.close(fig)

        # Get grid extents for plotting
        x_extent = [xi_np.min(), xi_np.max()]
        y_extent = [yi_np.min(), yi_np.max()]
        z_extent = [zi_np.min(), zi_np.max()]

        # Plot Center X-slice
        plot_slice_comparison(
            bIQ3_original_np[center_x_idx, :, :],
            bIQ3_fused_np[center_x_idx, :, :],
            f'Beamformed Output Comparison (Center X-slice at index {center_x_idx})',
            'comparison_center_x_slice.png',
            [y_extent[0], y_extent[1], z_extent[0], z_extent[1]], # Extent is [xmin, xmax, ymin, ymax] for imshow
            'Y (m)', 'Z (m)'
        )

        # Plot Center Y-slice
        plot_slice_comparison(
            bIQ3_original_np[:, center_y_idx, :],
            bIQ3_fused_np[:, center_y_idx, :],
            f'Beamformed Output Comparison (Center Y-slice at index {center_y_idx})',
            'comparison_center_y_slice.png',
            [x_extent[0], x_extent[1], z_extent[0], z_extent[1]], # Extent is [xmin, xmax, ymin, ymax] for imshow
            'X (m)', 'Z (m)'
        )

        # Plot Center Z-slice
        plot_slice_comparison(
            bIQ3_original_np[:, :, center_z_idx],
            bIQ3_fused_np[:, :, center_z_idx],
            f'Beamformed Output Comparison (Center Z-slice at index {center_z_idx})',
            'comparison_center_z_slice.png',
            [x_extent[0], x_extent[1], y_extent[0], y_extent[1]], # Extent is [xmin, xmax, ymin, ymax] for imshow
            'X (m)', 'Y (m)'
        )

        print("Visual comparison plots saved.")

        # --- Prepare Stacked IQ Data for Throughput Test ---
        print("\n--- Preparing 1000 stacked IQ volumes ---")
        num_total_volumes = 1000
        batch_increment = 10
        # Repeat the single IQ volume 1000 times along a new batch dimension
        IQ_1000_tensor = IQ3_tensor.repeat(num_total_volumes, 1, 1) # Shape: (1000, num_samples, num_elements)
        print(f"Stacked IQ data prepared. Shape: {IQ_1000_tensor.shape}")

        # --- Throughput Test ---
        print(f"\n--- Running Throughput Test ({num_total_volumes} volumes in batches of {batch_increment}) ---")

        # Time Original Beamformer
        print("Timing original vectorized_beamform (PyTorch JIT)...")
        original_total_time = 0.0
        with torch.no_grad():
            for i in range(0, num_total_volumes, batch_increment):
                iq_batch = IQ_1000_tensor[i : i + batch_increment, :, :]
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                _ = original_vectorized_beamform(
                     iq_batch, xi_tensor, yi_tensor, zi_tensor, txdel3_tensor, element_pos_tensor,
                     fs, fc, num_elements, c=speed_of_sound, device=device
                )
                end_event.record()
                torch.cuda.synchronize() # Wait for the batch to complete
                original_total_time += start_event.elapsed_time(end_event) # Time in ms

        original_avg_time_per_volume_ms = original_total_time / num_total_volumes
        original_throughput_vps = num_total_volumes / (original_total_time / 1000.0) # Convert ms to seconds

        print(f"Original Beamformer Total Time: {original_total_time:.4f} ms")
        print(f"Original Beamformer Average Time per Volume: {original_avg_time_per_volume_ms:.4f} ms")
        print(f"Original Beamformer Throughput: {original_throughput_vps:.2f} volumes/second")


        # Time Fused Extension Beamformer
        print("\nTiming fused vectorized_beamform_ext (PyTorch C++ Extension)...")
        fused_total_time = 0.0
        with torch.no_grad():
            for i in range(0, num_total_volumes, batch_increment):
                iq_batch = IQ_1000_tensor[i : i + batch_increment, :, :]
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                _ = fused_vectorized_beamform(
                    iq_batch, xi_tensor, yi_tensor, zi_tensor, txdel3_tensor, element_pos_tensor,
                    fs, fc, num_elements, c=speed_of_sound, device=device
                )
                end_event.record()
                torch.cuda.synchronize() # Wait for the batch to complete
                fused_total_time += start_event.elapsed_time(end_event) # Time in ms

        fused_avg_time_per_volume_ms = fused_total_time / num_total_volumes
        fused_throughput_vps = num_total_volumes / (fused_total_time / 1000.0) # Convert ms to seconds

        print(f"Fused Extension Beamformer Total Time: {fused_total_time:.4f} ms")
        print(f"Fused Extension Beamformer Average Time per Volume: {fused_avg_time_per_volume_ms:.4f} ms")
        print(f"Fused Extension Beamformer Throughput: {fused_throughput_vps:.2f} volumes/second")

        # --- Summary ---
        print("\n--- Comparison Summary ---")
        print(f"Original Beamformer Throughput: {original_throughput_vps:.2f} volumes/second")
        print(f"Fused Extension Beamformer Throughput: {fused_throughput_vps:.2f} volumes/second")
        if fused_throughput_vps > original_throughput_vps:
            print(f"The Fused Extension is approximately {fused_throughput_vps / original_throughput_vps:.2f}x faster than the Original.")
        elif original_throughput_vps > fused_throughput_vps:
             print(f"The Original is approximately {original_throughput_vps / fused_throughput_vps:.2f}x faster than the Fused Extension.")
        else:
            print("The throughput is approximately the same for both implementations.")


    except FileNotFoundError:
        print(f"Error: Data files not found in '{data_dir}'. Please run simulate_data.py first to generate the necessary data.")
    except ImportError:
         print("Error: Required libraries (numpy, torch, matplotlib, pymust, fused_beamform_ext) not found. Please install them and ensure the C++ extension is built.")
    except Exception as e:
        print(f"An error occurred during the comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_comparison()