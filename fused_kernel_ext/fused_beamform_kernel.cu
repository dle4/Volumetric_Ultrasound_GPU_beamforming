// fused_beamform_kernel.cu
#include <cuda_runtime.h>
#include <math_constants.h> // For CUDART_PI_F
#include <c10/cuda/CUDAStream.h> // For accessing CUDA stream from PyTorch
// #include <cstdio> // For printf - Removed for debugging
// #include <cmath> // For fabsf - Removed for debugging

// Use c10::complex<float> for complex numbers with PyTorch
#include <c10/util/complex.h>

// Define the CUDA kernel
__global__ void fused_beamform_kernel(
    const c10::complex<float>* __restrict__ iq_data,         // Input IQ data (Layout: [num_elements][num_samples])
    const float*  __restrict__ grid_x,          // X-coordinates for the current chunk (flattened)
    const float*  __restrict__ grid_y,          // Y-coordinates for the current chunk (flattened)
    const float*  __restrict__ grid_z,          // Z-coordinates for the current chunk (flattened)
    const float*  __restrict__ element_x,       // Element X positions (num_elements)
    const float*  __restrict__ element_y,       // Element Y positions (num_elements)
    const float*  __restrict__ element_z,       // Element Z positions (num_elements)
    const float*  __restrict__ tx_delays,       // Transmit delays (num_elements)

    c10::complex<float>*       __restrict__ beamformed_iq,   // Output beamformed IQ data for the chunk (flattened)

    float fs,                   // Sampling frequency
    float fc,                   // Center frequency
    float c,                    // Speed of sound
    int   num_samples,          // Number of samples per element in IQ data
    int   num_elements,         // Number of transducer elements
    int   num_chunk_points      // Total number of points in the current chunk
) {
    // 1. Thread Indexing & Bounds Check
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= num_chunk_points) return;

    // Define debug point based on coordinates near the scatterer (0, 0, 30e-3) - Removed for debugging
    // const float DEBUG_PX = 0.0f;
    // const float DEBUG_PY = 0.0f;
    // const float DEBUG_PZ = 30e-3f;
    // const float COORD_TOLERANCE = 1e-3f; // Increased tolerance to 1 mm

    // 2. Load Grid Point Coordinates for this Thread
    float px = grid_x[point_idx];
    float py = grid_y[point_idx];
    float pz = grid_z[point_idx];

    // bool is_debug_thread = (fabsf(px - DEBUG_PX) < COORD_TOLERANCE &&
    //                         fabsf(py - DEBUG_PY) < COORD_TOLERANCE &&
    //                         fabsf(pz - DEBUG_PZ) < COORD_TOLERANCE);

    // if (is_debug_thread) {
    //     printf("Debug Thread %d: Grid Point (%.6f, %.6f, %.6f)\n", point_idx, px, py, pz);
    // }

    // 3. Calculate Minimum Transmit Distance Time (dTX / c)
    float min_combined_dist = CUDART_INF_F;
    for (int elem = 0; elem < num_elements; ++elem) {
        float ex = element_x[elem];
        float ey = element_y[elem];
        float ez = element_z[elem];
        float dx = px - ex;
        float dy = py - ey;
        float dz = pz - ez;
        float receive_dist = sqrtf(dx * dx + dy * dy + dz * dz); // dRX
        float combined_dist = tx_delays[elem] * c + receive_dist; // delaysTX*c + dRX
        min_combined_dist = fminf(min_combined_dist, combined_dist);
    }
    float transmit_delay_time = min_combined_dist / c; // dTX / c

    // if (is_debug_thread) {
    //      printf("Debug Thread %d: Transmit Delay Time (dTX/c): %.9f\n", point_idx, transmit_delay_time);
    // }


    // 4. Core Calculation Loop (Iterate through Elements)
    c10::complex<float> accumulated_iq(0.0f, 0.0f);
    float wc = 2.0f * CUDART_PI_F * fc;

    // Define debug element (e.g., element 500) - Removed for debugging
    // const int DEBUG_ELEM = 499; // Index 499 for element 500

    for (int elem = 0; elem < num_elements; ++elem) {
        // a. Load Element Coords
        float ex = element_x[elem];
        float ey = element_y[elem];
        float ez = element_z[elem];

        // if (is_debug_thread && elem == DEBUG_ELEM) {
        //      printf("Debug Thread %d, Elem %d: Element Coords Used (%.6f, %.6f, %.6f)\n", point_idx, elem, ex, ey, ez);
        // }


        // b. Calculate Receive Distance & Delay
        float dx = px - ex;
        float dy = py - ey;
        float dz = pz - ez;
        float receive_dist = sqrtf(dx * dx + dy * dy + dz * dz); // dRX
        float receive_delay_time = receive_dist / c; // dRX / c

        // c. Calculate Total Delay
        float total_delay = transmit_delay_time + receive_delay_time;

        // d. Convert Delay to Sample Index
        float sample_idx_f = total_delay * fs;

        // e. Interpolation Indices & Weights
        int   idx0 = floorf(sample_idx_f);
        int   idx1 = idx0 + 1;
        float w1 = sample_idx_f - (float)idx0;
        float w0 = 1.0f - w1;

        c10::complex<float> iq_val(0.0f, 0.0f);

        // f. Boundary Checks, Data Fetch & Interpolation
        if (idx0 >= 0 && idx1 < num_samples) {
            // Assumes iq_data layout: [element][sample]
            int base_idx = elem * num_samples;
            c10::complex<float> iq0 = iq_data[base_idx + idx0];
            c10::complex<float> iq1 = iq_data[base_idx + idx1];

            // Linear Interpolation: iq_val = iq0 * w0 + iq1 * w1
            iq_val = iq0 * w0 + iq1 * w1;

            // g. Apodization (Implicit via bounds check, add explicit if needed)

            // h. Phase Rotation
            float angle = wc * total_delay;
            c10::complex<float> rotation_factor(cosf(angle), sinf(angle));

            // Apply rotation: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            iq_val = iq_val * rotation_factor;

            // i. Summation
            accumulated_iq += iq_val;

            // if (is_debug_thread && elem == DEBUG_ELEM) {
            //      printf("Debug Thread %d, Elem %d: receive_dist=%.9f, receive_delay_time=%.9f\n", point_idx, elem, receive_dist, receive_delay_time);
            //      printf("Debug Thread %d, Elem %d: total_delay=%.9f, sample_idx_f=%.9f\n", point_idx, elem, total_delay, sample_idx_f);
            //      printf("Debug Thread %d, Elem %d: idx0=%d, idx1=%d, w0=%.6f, w1=%.6f\n", point_idx, elem, idx0, idx1, w0, w1);
            //      printf("Debug Thread %d, Elem %d: iq0=(%.6f, %.6f), iq1=(%.6f, %.6f)\n", point_idx, elem, iq0.real(), iq0.imag(), iq1.real(), iq1.imag());
            //      printf("Debug Thread %d, Elem %d: interpolated_iq=(%.6f, %.6f)\n", point_idx, elem, iq_val.real(), iq_val.imag()); // After interpolation, before rotation
            //      printf("Debug Thread %d, Elem %d: angle=%.6f, rotation_factor=(%.6f, %.6f)\n", point_idx, elem, angle, rotation_factor.real(), rotation_factor.imag());
            //      printf("Debug Thread %d, Elem %d: rotated_iq=(%.6f, %.6f)\n", point_idx, elem, iq_val.real(), iq_val.imag()); // After rotation
            //      printf("Debug Thread %d, Elem %d: accumulated_iq=(%.6f, %.6f)\n", point_idx, elem, accumulated_iq.real(), accumulated_iq.imag());
            // }
        }
    }

    // 5. Write Output
    beamformed_iq[point_idx] = accumulated_iq;

    // if (is_debug_thread) {
    //     printf("Debug Thread %d: Final beamformed_iq=(%.6f, %.6f)\n", point_idx, beamformed_iq[point_idx].real(), beamformed_iq[point_idx].imag());
    // }
}

// C++ launcher function
void fused_beamform_cuda_launcher(
    const c10::complex<float>* iq_data,
    const float* grid_x,
    const float* grid_y,
    const float* grid_z,
    const float* element_x,
    const float* element_y,
    const float* element_z,
    const float* tx_delays,
    c10::complex<float>* beamformed_iq,
    float fs,
    float fc,
    float c,
    int num_samples,
    int num_elements,
    int num_chunk_points
) {
    // Define grid and block dimensions
    // Simple 1D grid, each thread processes one point in the chunk
    const int threads_per_block = 256; // Example block size
    const int blocks_per_grid = (num_chunk_points + threads_per_block - 1) / threads_per_block;

    // Get the current CUDA stream from PyTorch
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Launch the kernel
    fused_beamform_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        iq_data,
        grid_x,
        grid_y,
        grid_z,
        element_x,
        element_y,
        element_z,
        tx_delays,
        beamformed_iq,
        fs,
        fc,
        c,
        num_samples,
        num_elements,
        num_chunk_points
    );

    // Check for CUDA errors (optional but recommended for debugging)
    // CUDA_POST_KERNEL_CHECK; // Requires a macro or manual check
}