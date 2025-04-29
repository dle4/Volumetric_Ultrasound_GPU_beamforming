// fused_beamform_wrapper.cpp
#include <torch/extension.h>
#include <vector>

// Declare the CUDA kernel launcher function from fused_beamform_kernel.cu
// The function signature must match the launcher function in the .cu file.
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
);

// C++ function that will be called from Python
torch::Tensor fused_beamform_forward_cuda(
    torch::Tensor iq_data,         // Input IQ data (complex64) - Expected Layout: [num_elements][num_samples]
    torch::Tensor grid_x,          // X-coordinates (float32) - Expected Flattened Shape: [num_chunk_points]
    torch::Tensor grid_y,          // Y-coordinates (float32) - Expected Flattened Shape: [num_chunk_points]
    torch::Tensor grid_z,          // Z-coordinates (float32) - Expected Flattened Shape: [num_chunk_points]
    torch::Tensor element_x,       // Element X positions (float32) - Expected Shape: [num_elements]
    torch::Tensor element_y,       // Element Y positions (float32) - Expected Shape: [num_elements]
    torch::Tensor element_z,       // Element Z positions (float32) - Expected Shape: [num_elements]
    torch::Tensor tx_delays,       // Transmit delays (float32) - Expected Shape: [num_elements]
    float fs,                      // Sampling frequency
    float fc,                      // Center frequency
    float c                        // Speed of sound
) {
    // --- Input Validation ---
    TORCH_CHECK(iq_data.is_cuda(), "iq_data must be a CUDA tensor");
    TORCH_CHECK(grid_x.is_cuda(), "grid_x must be a CUDA tensor");
    TORCH_CHECK(grid_y.is_cuda(), "grid_y must be a CUDA tensor");
    TORCH_CHECK(grid_z.is_cuda(), "grid_z must be a CUDA tensor");
    TORCH_CHECK(element_x.is_cuda(), "element_x must be a CUDA tensor");
    TORCH_CHECK(element_y.is_cuda(), "element_y must be a CUDA tensor");
    TORCH_CHECK(element_z.is_cuda(), "element_z must be a CUDA tensor");
    TORCH_CHECK(tx_delays.is_cuda(), "tx_delays must be a CUDA tensor");

    TORCH_CHECK(iq_data.scalar_type() == torch::kComplexFloat, "iq_data must be complex64");
    TORCH_CHECK(grid_x.scalar_type() == torch::kFloat, "grid_x must be float32");
    TORCH_CHECK(grid_y.scalar_type() == torch::kFloat, "grid_y must be float32");
    TORCH_CHECK(grid_z.scalar_type() == torch::kFloat, "grid_z must be float32");
    TORCH_CHECK(element_x.scalar_type() == torch::kFloat, "element_x must be float32");
    TORCH_CHECK(element_y.scalar_type() == torch::kFloat, "element_y must be float32");
    TORCH_CHECK(element_z.scalar_type() == torch::kFloat, "element_z must be float32");
    TORCH_CHECK(tx_delays.scalar_type() == torch::kFloat, "tx_delays must be float32");

    TORCH_CHECK(grid_x.dim() == 1, "grid_x must be a 1D tensor (flattened)");
    TORCH_CHECK(grid_y.dim() == 1, "grid_y must be a 1D tensor (flattened)");
    TORCH_CHECK(grid_z.dim() == 1, "grid_z must be a 1D tensor (flattened)");
    TORCH_CHECK(element_x.dim() == 1, "element_x must be a 1D tensor");
    TORCH_CHECK(element_y.dim() == 1, "element_y must be a 1D tensor");
    TORCH_CHECK(element_z.dim() == 1, "element_z must be a 1D tensor");
    TORCH_CHECK(tx_delays.dim() == 1, "tx_delays must be a 1D tensor");
    TORCH_CHECK(iq_data.dim() == 2, "iq_data must be a 2D tensor ([elements][samples])");


    int num_elements = element_x.size(0);
    int num_samples = iq_data.size(1); // Assuming [elements][samples] layout
    int num_chunk_points = grid_x.size(0);

    TORCH_CHECK(element_y.size(0) == num_elements, "element_y size mismatch");
    TORCH_CHECK(element_z.size(0) == num_elements, "element_z size mismatch");
    TORCH_CHECK(tx_delays.size(0) == num_elements, "tx_delays size mismatch");
    TORCH_CHECK(iq_data.size(0) == num_elements, "iq_data element dimension size mismatch");
    TORCH_CHECK(grid_y.size(0) == num_chunk_points, "grid_y size mismatch");
    TORCH_CHECK(grid_z.size(0) == num_chunk_points, "grid_z size mismatch");


    // Ensure input tensors are contiguous (simplifies kernel indexing)
    TORCH_CHECK(iq_data.is_contiguous(), "iq_data must be contiguous");
    TORCH_CHECK(grid_x.is_contiguous(), "grid_x must be contiguous");
    TORCH_CHECK(grid_y.is_contiguous(), "grid_y must be contiguous");
    TORCH_CHECK(grid_z.is_contiguous(), "grid_z must be contiguous");
    TORCH_CHECK(element_x.is_contiguous(), "element_x must be contiguous");
    TORCH_CHECK(element_y.is_contiguous(), "element_y must be contiguous");
    TORCH_CHECK(element_z.is_contiguous(), "element_z must be contiguous");
    TORCH_CHECK(tx_delays.is_contiguous(), "tx_delays must be contiguous");


    // --- Allocate Output Tensor ---
    // Output shape is (num_chunk_points) for the flattened chunk
    torch::Tensor beamformed_iq = torch::empty({num_chunk_points}, iq_data.options());

    // --- Get Raw Data Pointers ---
    const c10::complex<float>* iq_data_ptr = iq_data.data_ptr<c10::complex<float>>();
    const float* grid_x_ptr = grid_x.data_ptr<float>();
    const float* grid_y_ptr = grid_y.data_ptr<float>();
    const float* grid_z_ptr = grid_z.data_ptr<float>();
    const float* element_x_ptr = element_x.data_ptr<float>();
    const float* element_y_ptr = element_y.data_ptr<float>();
    const float* element_z_ptr = element_z.data_ptr<float>();
    const float* tx_delays_ptr = tx_delays.data_ptr<float>();
    c10::complex<float>* beamformed_iq_ptr = beamformed_iq.data_ptr<c10::complex<float>>();

    // --- Launch CUDA Kernel ---
    fused_beamform_cuda_launcher(
        iq_data_ptr,
        grid_x_ptr,
        grid_y_ptr,
        grid_z_ptr,
        element_x_ptr,
        element_y_ptr,
        element_z_ptr,
        tx_delays_ptr,
        beamformed_iq_ptr,
        fs,
        fc,
        c,
        num_samples,
        num_elements,
        num_chunk_points
    );

    return beamformed_iq;
}

// --- Python Binding ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_beamform", &fused_beamform_forward_cuda, "Fused Beamform Forward (CUDA)");
}