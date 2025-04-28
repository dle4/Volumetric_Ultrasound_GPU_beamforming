import numpy as np
import os
import torch
import time
import pymust
from vectorized_beamformer3D import vectorized_beamform
import matplotlib.pyplot as plt
from pymust import utils

# Load simulated data
data_dir = "simulated_data"

try:
    print(f"Loading data from '{data_dir}'...")
    xi_np = np.load(os.path.join(data_dir, 'xi.npy'))
    yi_np = np.load(os.path.join(data_dir, 'yi.npy'))
    zi_np = np.load(os.path.join(data_dir, 'zi.npy'))
    txdel3_np = np.load(os.path.join(data_dir, 'txdel3.npy'))
    param_data = np.load(os.path.join(data_dir, 'param_data.npy'), allow_pickle=True).item()
    RF3_np = np.load(os.path.join(data_dir, 'RF3.npy'))
    print("Data loaded successfully.")

    # Demodulate RF data
    print("Demodulating RF data...")
    IQ3_np = pymust.rf2iq(RF3_np, param_data['fs'], param_data['fc'])
    print(f"I/Q data generated. Shape: {IQ3_np.shape}")

except FileNotFoundError:
    print(f"Error: Data files not found in '{data_dir}'. Please run simulate_data.py first.")
    exit()
except ImportError:
    print("Error: Required libraries (numpy, torch, pymust, matplotlib) not found. Please install them.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading or demodulation: {e}")
    exit()

# Extract parameters and convert all inputs to tensors
fs = param_data['fs']
fc = param_data['fc']
num_elements = param_data['Nelements']
element_pos_np = param_data['elements']
speed_of_sound = 1540.0 # Use default c

# Determine batch size and num_samples from the IQ data
if IQ3_np.ndim == 3:
    batch_size, num_samples, _ = IQ3_np.shape
elif IQ3_np.ndim == 2:
    num_samples, _ = IQ3_np.shape
    batch_size = 1 # Assuming single batch if no batch dim
    IQ3_np = IQ3_np[np.newaxis, :, :] # Add batch dim for consistency

# --- Beamforming and Timing ---

# 1. PyMUST dasmtx3 Beamforming (Keeping for comparison timing, but not plotting/comparing output)
print("\nRunning PyMUST dasmtx3 beamforming...")
start_time_pymust = time.perf_counter()
# Need to convert inputs to match original pymust.dasmtx3 expected types if necessary
# Assuming pymust.dasmtx3 expects numpy arrays and param object
param = utils.Param()
param.update(param_data)
M3 = pymust.dasmtx3(IQ3_np[0, :, :], xi_np, yi_np, zi_np, txdel3_np, param) # Get sparse matrix for single volume
bIQ3_pymust_np = pymust.utils.applyDasMTX(M3, IQ3_np[0, :, :], xi_np.shape) # Apply matrix and get dense array for single volume
end_time_pymust = time.perf_counter()
time_pymust = end_time_pymust - start_time_pymust
print(f"PyMUST beamforming complete (single volume). Shape: {bIQ3_pymust_np.shape}")
print(f"PyMUST time (single volume): {time_pymust:.4f} seconds")


# Convert inputs to tensors and move to device for PyTorch beamformer
IQ3_tensor = torch.from_numpy(IQ3_np).to(torch.complex64) # Keep on CPU for CPU test first
xi_tensor = torch.from_numpy(xi_np).to(torch.float32)
yi_tensor = torch.from_numpy(yi_np).to(torch.float32)
zi_tensor = torch.from_numpy(zi_np).to(torch.float32)
txdel3_tensor = torch.from_numpy(txdel3_np).to(torch.float32)
element_pos_tensor = torch.from_numpy(element_pos_np).to(torch.float32)


# 2. CPU PyTorch Beamforming
print("\nRunning CPU PyTorch beamforming...")
start_time_cpu = time.perf_counter()
bIQ3_cpu_tensor = vectorized_beamform(
    IQ3_tensor, xi_tensor, yi_tensor, zi_tensor, txdel3_tensor, element_pos_tensor,
    fs, fc, num_elements, c=speed_of_sound, device='cpu'
)
end_time_cpu = time.perf_counter()
time_cpu = end_time_cpu - start_time_cpu
bIQ3_cpu_np = bIQ3_cpu_tensor.cpu().numpy() # Convert to numpy for plotting
print(f"CPU PyTorch beamforming complete. Shape: {bIQ3_cpu_np.shape}")
print(f"CPU PyTorch time: {time_cpu:.4f} seconds")

# 3. GPU PyTorch Beamforming
bIQ3_gpu_np = None
time_gpu = None
if torch.cuda.is_available():
    device = 'cuda'
    print("\nRunning GPU PyTorch beamforming...")
    # Move tensors to GPU for GPU test
    IQ3_gpu_tensor = IQ3_tensor.to(device)
    xi_gpu_tensor = xi_tensor.to(device)
    yi_gpu_tensor = yi_tensor.to(device)
    zi_gpu_tensor = zi_tensor.to(device)
    txdel3_gpu_tensor = txdel3_tensor.to(device)
    element_pos_gpu_tensor = element_pos_tensor.to(device)


    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warm-up run
    _ = vectorized_beamform(
        IQ3_gpu_tensor, xi_gpu_tensor, yi_gpu_tensor, zi_gpu_tensor, txdel3_gpu_tensor, element_pos_gpu_tensor,
        fs, fc, num_elements, c=speed_of_sound, device=device
    )
    torch.cuda.synchronize() # Ensure warm-up is complete

    start_event.record()
    bIQ3_gpu_tensor = vectorized_beamform(
        IQ3_gpu_tensor, xi_gpu_tensor, yi_gpu_tensor, zi_gpu_tensor, txdel3_gpu_tensor, element_pos_gpu_tensor,
        fs, fc, num_elements, c=speed_of_sound, device=device
    )
    end_event.record()

    torch.cuda.synchronize() # Wait for all GPU ops and CPU transfer
    time_gpu = start_event.elapsed_time(end_event) / 1000.0 # Convert to seconds
    bIQ3_gpu_np = bIQ3_gpu_tensor.cpu().numpy() # Convert to numpy for plotting
    print(f"GPU PyTorch beamforming complete. Shape: {bIQ3_gpu_np.shape}")
    print(f"GPU PyTorch time: {time_gpu:.4f} seconds")
else:
    print("\nCUDA not available. Skipping GPU PyTorch beamforming.")

# --- Display Times ---
print("\n--- Beamformer Execution Times ---")
print(f"PyMUST time (single volume): {time_pymust:.4f} seconds")
print(f"CPU PyTorch time: {time_cpu:.4f} seconds")
if time_gpu is not None:
    print(f"GPU PyTorch time: {time_gpu:.4f} seconds")
    if time_cpu is not None and time_cpu > 0:
        speedup_gpu_vs_cpu = time_cpu / time_gpu
        print(f"GPU vs CPU Speedup: {speedup_gpu_vs_cpu:.2f}x")

# --- 2D Plotting of Center Slice ---
print("\n--- Generating 2D Plots of Center Slice ---")

def plot_center_slice(data_np, xi_np, yi_np, zi_np, title):
    if data_np is None:
        print(f"Skipping plot for '{title}' as data is not available.")
        return None

    # Find the index of the center slice in the z-dimension
    center_z_index = zi_np.shape[2] // 2

    # Extract the center slice
    slice_data = data_np[0, :, :, center_z_index] # Assuming batch size 1 for plotting
    slice_xi = xi_np[:, :, center_z_index]
    slice_yi = yi_np[:, :, center_z_index]

    # Calculate intensity in dB
    abs_data = np.abs(slice_data)
    abs_data[abs_data == 0] = np.finfo(float).eps # Avoid log10(0)
    db_data = 20 * np.log10(abs_data / np.max(abs_data))

    # Create plot
    fig, ax = plt.subplots()
    # Use extent to set the axis limits based on spatial coordinates
    im = ax.imshow(db_data, extent=[slice_xi.min(), slice_xi.max(), slice_yi.max(), slice_yi.min()], aspect='auto', cmap='gray')
    ax.set_title(title)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    plt.colorbar(im, label='Intensity (dB)')
    return fig

# Plot the center slice for each beamformer's output
# fig_pymust = plot_center_slice(bIQ3_pymust_np, xi_np, yi_np, zi_np, 'PyMUST Beamformed Output (Center Slice)') # Removed PyMUST plot
fig_cpu = plot_center_slice(bIQ3_cpu_np, xi_np, yi_np, zi_np, 'CPU PyTorch Beamformed Output (Center Slice)')
fig_gpu = plot_center_slice(bIQ3_gpu_np, xi_np, yi_np, zi_np, 'GPU PyTorch Beamformed Output (Center Slice)')

plt.tight_layout()
plt.show()

print("--- Plotting Complete ---")