# 3D GPU Beamformer

## Overview

This project implements vectorized 3D Delay-and-Sum beamformers utilizing PyTorch for execution on the GPU. It is designed to process I/Q data and reconstruct 3D volumes, with capabilities for chunking along the Z, X, and Y axes to manage memory usage and improve performance. Two main implementations are provided: a pure PyTorch/JIT version (`vectorized_beamformer3D.py`) and a version leveraging a fused CUDA kernel implemented as a PyTorch C++ Extension (`vectorized_beamformer3D_ext.py`).

## Development Process

This beamformer was developed and debugged iteratively. Initially, an issue where the beamformer only processed a single Z-line was identified and fixed by correctly accumulating results from Z-chunks. Subsequently, efforts focused on optimizing performance and addressing VRAM limitations. Internal XY chunking was implemented to process the reconstruction grid in smaller spatial blocks, which improved throughput and helped manage VRAM when processing batches of volumes. More recently, a fused CUDA kernel was developed and integrated via a PyTorch C++ extension to further enhance performance by reducing kernel launch overhead and improving data locality.

## Performance Optimization and Comparison

Significant effort has been made to optimize the performance of the beamformer on the GPU. Key optimizations include:

*   **Kernel Fusion with `torch.jit.script` (Original Implementation):** The core computation logic within each processing chunk in the original PyTorch implementation is compiled using `torch.jit.script`. This technique fuses multiple PyTorch operations into fewer, larger CUDA kernels, reducing kernel launch overhead and improving data locality.
*   **Fused CUDA Kernel (C++ Extension Implementation):** The `vectorized_beamformer3D_ext.py` implementation utilizes a custom fused CUDA kernel implemented in C++ and exposed to Python via a PyTorch extension. This approach allows for fine-grained control over memory access and computation, leading to potentially higher performance compared to the JIT-compiled PyTorch code.
*   **Optimal Chunking Strategy:** Through systematic benchmarking, optimal chunk sizes for processing the volumetric grid have been identified for the target hardware (NVIDIA RTX 4060 Ti) and data dimensions (IQ: 560x1024, Grid: 64x64x128). The optimal chunk sizes found are **X=8, Y=16, Z=8**. These chunk sizes are used in both implementations for a fair comparison.

A comparison script (`run_comparison.py`) was created to evaluate the performance of the two implementations using 1000 stacked IQ volumes processed in batches of 10. The results on an NVIDIA RTX 4060 Ti are as follows:

*   **Original PyTorch JIT Beamformer:** 3.84 volumes/second
*   **Fused C++ Extension Beamformer:** 5.71 volumes/second

The Fused C++ Extension implementation is approximately **1.49x faster** than the Original PyTorch JIT implementation for this specific test case.

## Required Packages

To run these beamformers and the associated scripts, you need the following Python packages:

*   `numpy`
*   `torch` (with CUDA support for GPU execution)
*   `pymust` (used in the example/test scripts for data loading and demodulation)
*   `matplotlib` (for visualization in the comparison script)

Additionally, the fused beamformer (`vectorized_beamformer3D_ext.py`) requires a compiled PyTorch C++ extension (`fused_beamform_ext`).

You can install most of these packages using pip:

```bash
pip install numpy torch pymust matplotlib
```

*(Note: Installing PyTorch with CUDA support may require specific instructions based on your system and CUDA version. Refer to the official PyTorch documentation for details.)*

## Building the C++ Extension

The `vectorized_beamformer3D_ext.py` beamformer relies on a custom fused CUDA kernel implemented in C++ and exposed to Python as a PyTorch extension. Before using `vectorized_beamformer3D_ext.py`, you must compile and build this extension.

Ensure you have a compatible C++ compiler (like g++ or MSVC) and the NVIDIA CUDA Toolkit installed and configured correctly for your system and PyTorch installation.

Navigate to the `GPU_volumetric_beamforming` directory in your terminal and run the `setup.py` script using the following command:

```bash
python setup.py install
```

or, for development mode (which symlinks the build directory to your site-packages, allowing for easier iteration on the C++ code without repeated installs):

```bash
python setup.py develop
```

This command will compile the C++ and CUDA source files and build the `fused_beamform_ext` Python module. If the compilation is successful, you should be able to import `fused_beamform_ext` in your Python scripts.

## PyMUST Conventions

This project aims to align with the data conventions used in the PyMUST toolbox where applicable. Specifically:

*   **IQ Data Format:** The input IQ data is expected to be in a format compatible with PyMUST's output, typically with dimensions corresponding to (number of samples, number of elements) or (batch size, number of samples, number of elements).
*   **Grid Coordinates:** The beamforming grid coordinates (`xi`, `yi`, `zi`) are structured similarly to how they might be defined or used within PyMUST for 3D reconstruction.
*   **Element Positions:** Transducer element positions (`element_pos`) are expected in a format consistent with PyMUST's representation, typically (2, number of elements) for 2D coordinates.
*   **Transmit Delays:** Transmit delays (`txdel`) are handled in a manner consistent with PyMUST's approach for plane wave or similar transmit schemes.

This alignment facilitates easier integration and comparison with workflows or data generated using the PyMUST toolbox.

## Usage

The core beamforming logic is implemented in the `vectorized_beamform` function in `vectorized_beamformer3D.py` and the `vectorized_beamform_ext` function in `vectorized_beamformer3D_ext.py`.

Both functions have similar signatures:

```python
def vectorized_beamform(iq_data: torch.Tensor, xi: torch.Tensor, yi: torch.Tensor, zi: torch.Tensor, txdel: torch.Tensor, element_pos: torch.Tensor, fs: float, fc: float, num_elements: int, c: float = 1540.0, device: str = 'cuda', z_chunk_size: int = 8, x_chunk_size: int = 8, y_chunk_size: int = 16):
    """
    Performs vectorized Delay-and-Sum beamforming on I/Q data using PyTorch,
    with chunking along the Z-axis and optionally along X and Y axes to manage memory.
    Calls a JIT-scripted function for chunk processing (original) or a fused CUDA kernel (extension).
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
    # The actual function implementation details are in vectorized_beamformer3D.py and vectorized_beamformer3D_ext.py
    pass

```

Key parameters for performance and memory management include:

*   `device`: Specifies the device for computation ('cpu' or 'cuda').
*   `z_chunk_size`: Number of Z-slices processed in each chunk along the Z-axis.
*   `x_chunk_size`, `y_chunk_size`: Number of X and Y points processed in each chunk within a Z-slice.

## Testing and Throughput Measurement

The `run_comparison.py` script is provided to perform both visual and throughput comparisons between the two beamformer implementations.

To run the comparison:

```bash
python GPU_volumetric_beamforming/run_comparison.py
```

This script will:
*   Load necessary data from the `simulated_data/` directory.
*   Generate visual comparison plots of the center X, Y, and Z slices for a single volume, saved to the `example_outputs/` directory.
*   Measure and print the throughput (volumes per second) for both beamformers processing 1000 stacked IQ volumes.

The original `vectorized_beamformer3D.py` script also includes an example usage block (`if __name__ == '__main__':`) that demonstrates how to load data, run the beamformer, and measure its execution time for a single volume.


## Future Work

*   Further investigate potential numerical differences between the two implementations if accuracy requirements is stricter.
*   Explore further performance optimizations, such as manual mixed precision for specific kernels or alternative CUDA kernel implementations.
*   Add more comprehensive unit tests.
*   Improve the C++ extension build process documentation.
