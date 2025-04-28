# 3D GPU Beamformer

## Overview

This project implements a vectorized 3D Delay-and-Sum beamformer utilizing PyTorch for execution on the GPU. It is designed to process I/Q data and reconstruct 3D volumes, with capabilities for chunking along the Z, X, and Y axes to manage memory usage and improve performance.

This work uses PyMust as a reference for generating simulation data and parameter inputs. 

https://github.com/creatis-ULTIM/PyMUST

## Development Process

This beamformer was developed and debugged iteratively. Initially, an issue where the beamformer only processed a single Z-line was identified and fixed by correctly accumulating results from Z-chunks. Subsequently, efforts focused on optimizing performance and addressing VRAM limitations. Internal XY chunking was implemented to process the reconstruction grid in smaller spatial blocks, which improved throughput and helped manage VRAM when processing batches of volumes.

## Performance Optimization

Significant effort has been made to optimize the performance of the beamformer on the GPU. Key optimizations include:

*   **Kernel Fusion with `torch.jit.script`:** The core computation logic within each processing chunk has been compiled using `torch.jit.script`. This technique fuses multiple PyTorch operations into fewer, larger CUDA kernels, reducing kernel launch overhead and improving data locality.
*   **Optimal Chunking Strategy:** Through systematic benchmarking, optimal chunk sizes for processing the volumetric grid have been identified for the target hardware (NVIDIA RTX 4060 Ti) and data dimensions (IQ: 560x1024, Grid: 64x64x128). The optimal chunk sizes found are **X=8, Y=16, Z=8**.

These optimizations have resulted in a significant speedup compared to the initial implementation. The beamforming time for a single volume on the NVIDIA RTX 4060 Ti with optimal chunking and JIT compilation is approximately **0.3090 seconds**, compared to a baseline of 1.4683 seconds without these optimizations.

Automatic mixed precision (`torch.cuda.amp.autocast`) was also explored but did not yield a performance improvement in initial tests for this specific workload.

## Required Packages

To run this beamformer and the associated test script, you need the following Python packages:

*   `numpy`
*   `torch` (with CUDA support for GPU execution)
*   `pymust` (used in the example/test scripts for data loading and demodulation)
*   `tensorboard` and `tensorboard-plugin-profile` (for performance profiling)

You can install these packages using pip:

```bash
pip install numpy torch pymust tensorboard tensorboard-plugin-profile
```

*(Note: Installing PyTorch with CUDA support may require specific instructions based on your system and CUDA version. Refer to the official PyTorch documentation for details.)*

## Usage

The core beamforming logic is implemented in the `vectorized_beamform` function in `vectorized_beamformer3D.py`.

```python
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
    # ... function implementation details are in vectorized_beamformer3D.py
    pass # This is a placeholder for the actual function code in the .py file

```

Key parameters for performance and memory management include:

*   `device`: Specifies the device for computation ('cpu' or 'cuda').
*   `z_chunk_size`: Number of Z-slices processed in each chunk along the Z-axis (defaulting to 8 for optimal performance).
*   `x_chunk_size`, `y_chunk_size`: Number of X and Y points processed in each chunk within a Z-slice (defaulting to 8 and 16 respectively for optimal performance).

## Testing and Throughput Measurement

The `vectorized_beamformer3D.py` script includes an example usage block (`if __name__ == '__main__':`) that demonstrates how to load data, run the beamformer, and measure its execution time.

To run the example:

```bash
python GPU_volumetric_beamforming/vectorized_beamformer3D.py
```



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

*   Further investigate potential numerical differences between this implementation and other beamforming methods if accuracy requirements are stricter.
*   Explore further performance optimizations, such as manual mixed precision for specific kernels or custom CUDA kernels, if even greater speedup is required.
*   Add more comprehensive unit tests.
