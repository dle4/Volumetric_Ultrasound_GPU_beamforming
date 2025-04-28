# 3D GPU Beamformer

## Overview

This project implements a vectorized 3D Delay-and-Sum beamformer utilizing PyTorch for execution on the GPU. It is designed to process I/Q data and reconstruct 3D volumes, with capabilities for chunking along the Z, X, and Y axes to manage memory usage and improve performance.

## Development Process

This beamformer was developed and debugged iteratively. Initially, an issue where the beamformer only processed a single Z-line was identified and fixed by correctly accumulating results from Z-chunks. Subsequently, efforts focused on optimizing performance and addressing VRAM limitations. Internal XY chunking was implemented to process the reconstruction grid in smaller spatial blocks, which improved throughput and helped manage VRAM when processing batches of volumes.

## Required Packages

To run this beamformer and the associated test script, you need the following Python packages:

*   `numpy`
*   `torch` (with CUDA support for GPU execution)
*   `pymust` (used in the example/test scripts for data loading and demodulation)

You can install these packages using pip:

```bash
pip install numpy torch pymust
```

*(Note: Installing PyTorch with CUDA support may require specific instructions based on your system and CUDA version. Refer to the official PyTorch documentation for details.)*

## Usage

The core beamforming logic is implemented in the `vectorized_beamform` function in `vectorized_beamformer3D.py`.

```python
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
    # ... function implementation details are in vectorized_beamformer3D.py
    pass # This is a placeholder for the actual function code in the .py file

```

Key parameters for performance and memory management include:

*   `device`: Specifies the device for computation ('cpu' or 'cuda').
*   `z_chunk_size`: Number of Z-slices processed in each chunk along the Z-axis.
*   `x_chunk_size`, `y_chunk_size`: Number of X and Y points processed in each chunk within a Z-slice (set to -1 for no chunking along that dimension).
*   `input_on_gpu`, `output_on_gpu`: Flags to indicate if input/output tensors are already/should remain on the GPU.

## Testing and Throughput Measurement

The `gpu_streaming_throughput_test.py` script provides an example of how to use the `vectorized_beamform` function and measure its throughput. The current version of this script is configured to test processing a batch of 2 frames with internal XY (quadrant) and Z (size 1) chunking.

To run the throughput test:

```bash
python gpu_streaming_throughput_test.py
```

The script will report the execution time for the batch and the calculated throughput in frames per second. It also performs an accuracy check against a reference volume.

## Packaging for GitHub

To package this project for GitHub, ensure you have the following files in your repository:

*   `vectorized_beamformer3D.py`
*   `gpu_streaming_throughput_test.py`
*   `README.md` (this file)
*   `requirements.txt` (listing required packages)
*   Any necessary data files (e.g., in a `simulated_data` directory, as used in the test script).

Create a `requirements.txt` file with the following content:

```
numpy
torch
pymust
```

You can then initialize a Git repository, add these files, and push them to your GitHub repository.

## Future Work

*   Investigate and fix the accuracy discrepancy with the PyMUST reference.
*   Further optimize performance by experimenting with different chunk sizes, exploring kernel fusion (e.g., using `torch.jit.script`), or considering mixed precision if applicable and numerically stable.
*   Add more comprehensive unit tests.
