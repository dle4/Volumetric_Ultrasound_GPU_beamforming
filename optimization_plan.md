# GPU Volumetric Beamformer Optimization Plan

**Goal:** Significantly improve the execution speed of the GPU volumetric beamformer (`vectorized_beamformer3D.py`) on an NVIDIA RTX 4060 Ti by exploring mixed-precision (FP16/BF16) and kernel fusion techniques, while maintaining acceptable numerical accuracy.

**Target Data & Hardware:**

*   **IQ Data Shape:** (560, 1024)
*   **Grid Dimensions (Nx, Ny, Nz):** (64, 64, 128)
*   **GPU:** NVIDIA RTX 4060 Ti
*   **Accuracy:** Lower precision acceptable if speedup is significant.

**Assumptions & Prerequisites:**

1.  **Input Data:** Simulated or real data matching the target dimensions is available (e.g., in `simulated_data/`).
2.  **Environment:** Python environment with PyTorch (CUDA-enabled), NumPy, and optionally TensorBoard (`pip install tensorboard`) is set up.

**Optimization Strategies to Explore:**

1.  **Mixed Precision:** Utilize FP16 or BF16 for calculations and/or data storage to leverage Tensor Cores and reduce memory bandwidth.
2.  **Kernel Fusion:** Combine multiple small GPU operations into fewer, larger ones using `torch.jit.script` or custom kernels to reduce launch overhead.

**Proposed Plan & Testing Phases:**

```mermaid
graph TD
    A[Phase 1: Baseline Analysis] --> B(Phase 2: Mixed Precision);
    A --> C(Phase 3: Kernel Fusion);
    B --> D{Combine Best Techniques};
    C --> D;
    D --> E[Phase 4: Combined Optimization & Benchmarking];
    E --> F[Phase 5: Documentation];

    subgraph Phase 1: Baseline Performance Analysis
        A1[Configure/Run torch.profiler];
        A2[Analyze Bottlenecks (Time/Memory)];
        A3[Record Baseline Metrics];
    end

    subgraph Phase 2: Mixed Precision Optimization
        B1[Sub-Phase 2a: Test torch.amp];
        B2[Measure Speed & Accuracy (AMP)];
        B3[Sub-Phase 2b: Test Manual Casting (Optional)];
        B4[Measure Speed & Accuracy (Manual)];
    end

    subgraph Phase 3: Kernel Fusion Optimization
        C1[Sub-Phase 3a: Test torch.jit.script];
        C2[Measure Speed & Correctness (JIT)];
        C3[Sub-Phase 3b: Custom Kernels (Advanced/Optional)];
        C4[Measure Speed & Correctness (Custom)];
    end

    subgraph Phase 4: Combined Optimization & Benchmarking
        D1[Integrate Best Mixed Precision];
        D2[Integrate Best Fusion];
        E1[Profile Combined Approach];
        E2[Benchmark vs Baseline (Vary Chunk Sizes)];
        E3[Determine Optimal Configuration];
    end

     subgraph Phase 5: Documentation & Reporting
        F1[Summarize Speedup & Accuracy];
        F2[Document Code Changes];
        F3[Update README/Report];
    end
```

**Detailed Steps per Phase:**

1.  **Phase 1: Baseline Performance Analysis**
    *   **Goal:** Establish current performance and identify bottlenecks.
    *   **Actions:**
        *   Verify `torch.profiler` setup in `vectorized_beamformer3D.py`.
        *   Run the script with target data (IQ: 560x1024, Grid: 64x64x128) on the RTX 4060 Ti. Use a fixed, reasonable chunk size (e.g., `z_chunk_size=4`).
        *   Analyze profiler output (e.g., `tensorboard --logdir ./log`) for GPU kernel times, CPU vs GPU time, and memory operations.
        *   Record the total execution time as the baseline.
        *   Save the baseline beamformed output for accuracy comparisons.

2.  **Phase 2: Mixed-Precision Optimization**
    *   **Goal:** Evaluate speedup and accuracy using FP16/BF16.
    *   **Actions (Iterative Testing):**
        *   **Test `torch.amp`:** Wrap the main calculation loop with `torch.cuda.amp.autocast(dtype=torch.float16)`. Profile, measure time, calculate speedup, compare output accuracy.
        *   **Test Manual Casting (Optional):** Manually cast specific tensors (e.g., delays, interpolated samples) to `.half()` or `.bfloat16()`. Profile, measure time, check accuracy.

3.  **Phase 3: Kernel Fusion Optimization**
    *   **Goal:** Evaluate speedup from reducing kernel launches.
    *   **Actions (Iterative Testing):**
        *   **Test `torch.jit.script`:** Apply `@torch.jit.script` to the `vectorized_beamform` function or inner loop. Verify correctness, profile, measure time, calculate speedup.
        *   **Explore Custom Kernels (Advanced/Optional):** If needed, write custom CUDA kernels (CuPy, Numba, C++) to fuse operations like delay calculation, interpolation, and phase rotation. Implement, profile, measure time, check accuracy.

4.  **Phase 4: Combined Optimization & Benchmarking**
    *   **Goal:** Integrate the most promising techniques and find the best parameters.
    *   **Actions:**
        *   Create a version combining the best mixed-precision and fusion approaches.
        *   Verify correctness and accuracy.
        *   Profile the combined version.
        *   Systematically benchmark by varying chunk sizes (`z_chunk_size`, `x_chunk_size`, `y_chunk_size`) to find the optimal configuration. Compare against the baseline.

5.  **Phase 5: Documentation & Reporting**
    *   **Goal:** Document the results and final approach.
    *   **Actions:**
        *   Summarize final speedup and numerical accuracy.
        *   Document code modifications.
        *   Recommend optimal chunk sizes.
        *   Update `README.md` or create a report.