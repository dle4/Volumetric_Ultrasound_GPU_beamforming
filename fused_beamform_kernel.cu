// fused_beamform_kernel.cu
#include <cuda_runtime.h>
#include <math_constants.h> // For CUDART_PI_F

// Use float2 for complex numbers (real, imag)

extern "C" __global__ void fused_beamform_kernel(
    // Input Buffers (Pointers to GPU memory)
    const float2* __restrict__ iq_data,         // Input IQ data (Layout: [num_elements][num_samples])
    const float*  __restrict__ grid_x,          // X-coordinates for the current chunk (flattened)
    const float*  __restrict__ grid_y,          // Y-coordinates for the current chunk (flattened)
    const float*  __restrict__ grid_z,          // Z-coordinates for the current chunk (flattened)
    const float*  __restrict__ element_x,       // Element X positions (num_elements)
    const float*  __restrict__ element_y,       // Element Y positions (num_elements)
    const float*  __restrict__ element_z,       // Element Z positions (num_elements)
    const float*  __restrict__ tx_delays,       // Transmit delays (num_elements)

    // Output Buffer (Pointer to GPU memory)
    float2*       __restrict__ beamformed_iq,   // Output beamformed IQ data for the chunk (flattened)

    // Scalar Parameters
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

    // 2. Load Grid Point Coordinates for this Thread
    float px = grid_x[point_idx];
    float py = grid_y[point_idx];
    float pz = grid_z[point_idx];

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

    // 4. Core Calculation Loop (Iterate through Elements)
    float2 accumulated_iq = make_float2(0.0f, 0.0f);
    float wc = 2.0f * CUDART_PI_F * fc;

    for (int elem = 0; elem < num_elements; ++elem) {
        // a. Load Element Coords
        float ex = element_x[elem];
        float ey = element_y[elem];
        float ez = element_z[elem];

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

        float2 iq_val = make_float2(0.0f, 0.0f);

        // f. Boundary Checks, Data Fetch & Interpolation
        if (idx0 >= 0 && idx1 < num_samples) {
            // Assumes iq_data layout: [element][sample]
            int base_idx = elem * num_samples;
            float2 iq0 = iq_data[base_idx + idx0];
            float2 iq1 = iq_data[base_idx + idx1];
            iq_val.x = iq0.x * w0 + iq1.x * w1;
            iq_val.y = iq0.y * w0 + iq1.y * w1;

            // g. Apodization (Implicit via bounds check, add explicit if needed)

            // h. Phase Rotation
            float angle = wc * total_delay;
            float2 rotation_factor = make_float2(cosf(angle), sinf(angle));
            float real_part = iq_val.x * rotation_factor.x - iq_val.y * rotation_factor.y;
            float imag_part = iq_val.x * rotation_factor.y + iq_val.y * rotation_factor.x;
            iq_val = make_float2(real_part, imag_part);

            // i. Summation
            accumulated_iq.x += iq_val.x;
            accumulated_iq.y += iq_val.y;
        }
    }

    // 5. Write Output
    beamformed_iq[point_idx] = accumulated_iq;
}