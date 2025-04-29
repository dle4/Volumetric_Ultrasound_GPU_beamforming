# Plan: Compare Vectorized Beamformers (PyTorch JIT vs. C++ Extension)

This plan outlines the steps to compare the performance and output of `vectorized_beamformer3D.py` (pure PyTorch/JIT) and `vectorized_beamformer3D_ext.py` (using a C++/CUDA extension).

**Objective:**

*   Visually compare the beamformed output (center X, Y, Z slices) from both implementations using the same input data.
*   Measure and compare the processing throughput (volumes per second) of both implementations when processing 1000 identical IQ volumes in batches of 10.

**Key Requirement:** Ensure both beamformers use the *exact same* transducer element positions loaded from `simulated_data/param_data.npy`.

**Phase 1: Modify the C++ Extension Wrapper (`vectorized_beamformer3D_ext.py`)**

1.  **Goal:** Update the Python wrapper to accept `element_pos` as input instead of generating it internally.
2.  **Action:**
    *   Modify the `vectorized_beamform_ext` function signature to include `element_pos: torch.Tensor`.
    *   Remove the internal code that generates `element_x/y/z_tensor` based on a fixed 32x32 grid assumption.
    *   Add code to process the input `element_pos` tensor:
        *   Ensure it's on the correct `device`.
        *   Extract X and Y coordinates (assuming `element_pos` shape is `[2, num_elements]`).
        *   Create a Z coordinate tensor filled with zeros.
        *   Flatten X, Y, and Z coordinates in C-order (row-major) to create `element_x_tensor`, `element_y_tensor`, `element_z_tensor` suitable for the `fused_beamform_ext.fused_beamform` function.
    *   Update the `if __name__ == '__main__':` block to load `element_pos` from `param_data.npy` and pass this tensor to the modified `vectorized_beamform_ext` function.

**Phase 2: Create the Comparison Script (`GPU_volumetric_beamforming/run_comparison.py`)**

1.  **Goal:** Create a new Python script that performs both the visual and throughput comparisons.
2.  **Action:** This script will:
    *   **Imports:** Import necessary libraries (`numpy`, `torch`, `time`, `os`, `matplotlib.pyplot`, `pymust`) and the two beamforming functions (`vectorized_beamform` from the original file, and the *modified* `vectorized_beamform_ext` from the extension file).
    *   **Setup:** Check for CUDA, define data/output directories.
    *   **Load Data:** Load simulation parameters (`param_data`), grid coordinates (`xi`, `yi`, `zi`), transmit delays (`txdel3`), element positions (`element_pos`), and the raw RF data (`RF3`) from the `simulated_data` directory.
    *   **Prepare Single IQ Volume:** Demodulate the loaded `RF3` using `pymust.rf2iq` to get a single `IQ3_np` volume. Convert all necessary NumPy arrays (IQ, grid, delays, element positions) to PyTorch tensors on the CUDA device. Add a batch dimension of 1 to the single IQ tensor (`IQ3_tensor`).
    *   **Visual Comparison:**
        *   Run both `vectorized_beamform` and `vectorized_beamform_ext` on the single `IQ3_tensor`.
        *   Calculate the center indices for X, Y, and Z dimensions of the output volume.
        *   Generate three separate figures using `matplotlib`:
            *   Figure 1: Center X-slice comparison (Original vs. Fused Extension).
            *   Figure 2: Center Y-slice comparison (Original vs. Fused Extension).
            *   Figure 3: Center Z-slice comparison (Original vs. Fused Extension).
        *   Save these figures to an `output_dir`.
    *   **Prepare Stacked IQ Data:** Create `IQ_1000_tensor` by repeating the single `IQ3_tensor` (without the batch dim) 1000 times along a new batch dimension (final shape: `[1000, num_samples, num_elements]`).
    *   **Throughput Test:**
        *   Define `num_total_volumes = 1000` and `batch_increment = 10`.
        *   **Time Original:** Loop through `IQ_1000_tensor` in steps of `batch_increment`. For each batch, record the execution time of `vectorized_beamform` using `torch.cuda.Event` for accurate GPU timing. Sum the times.
        *   **Time Fused Extension:** Repeat the timing loop, but call the modified `vectorized_beamform_ext`. Sum the times.
        *   **Calculate & Print:** Calculate average throughput (volumes per second) for both methods based on the total time and `num_total_volumes`. Print the results clearly.

**Phase 3: Mermaid Diagram of the Plan**

```mermaid
graph TD
    A[Start Comparison Task] --> B{Phase 1: Modify Extension Wrapper};
    B --> B1[Modify vectorized_beamform_ext signature];
    B1 --> B2[Remove internal element generation];
    B2 --> B3[Add code to process input element_pos];
    B3 --> B4[Update example usage in ext.py];
    B4 --> C{Phase 2: Create Comparison Script};
    C --> C1[Imports & Setup];
    C1 --> C2[Load Data & Prepare Single IQ Tensor];
    C2 --> C3[Visual Comparison (Single Volume)];
    C3 --> C3a[Run Original Beamformer];
    C3 --> C3b[Run Fused Extension Beamformer];
    C3a & C3b --> C3c[Calculate Center Slices];
    C3c --> C3d[Generate & Save Comparison Plots (X, Y, Z)];
    C3d --> C4[Prepare Stacked IQ Data (1000 Volumes)];
    C4 --> C5[Throughput Test (Batches of 10)];
    C5 --> C5a[Time Original Beamformer Loop];
    C5 --> C5b[Time Fused Extension Beamformer Loop];
    C5a & C5b --> C6[Calculate & Print Throughput Results];
    C6 --> D[End];

    style B fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#ccf,stroke:#333,stroke-width:2px