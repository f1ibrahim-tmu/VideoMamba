#include "rk_discretize.h"

// Implementation of utility functions
void check_cuda(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "Tensor must be on CUDA device");
}

void check_contiguous(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
}

void check_dim(const torch::Tensor& tensor, int64_t dim) {
    TORCH_CHECK(tensor.dim() == dim, 
               "Expected ", dim, "-dimensional tensor, but got ", tensor.dim(), "-dimensional tensor");
}

// Python-accessible function implementations
torch::Tensor rk4_discretize(
    torch::Tensor state,
    torch::Tensor input_x,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor dt) {
    
    // Input validation
    check_cuda(state);
    check_cuda(input_x);
    check_cuda(A);
    check_cuda(B);
    check_cuda(dt);
    
    check_contiguous(state);
    check_contiguous(input_x);
    check_contiguous(A);
    check_contiguous(B);
    check_contiguous(dt);
    
    check_dim(state, 3);  // [batch, d_inner, d_state]
    check_dim(input_x, 2);  // [batch, d_inner]
    check_dim(A, 2);  // [d_inner, d_state]
    check_dim(B, 2);  // [d_inner, d_state]
    check_dim(dt, 2);  // [batch, d_inner]
    
    return rk4_discretize_cuda(state, input_x, A, B, dt);
}

std::vector<torch::Tensor> rk4_discretize_backward(
    torch::Tensor grad_output,
    torch::Tensor state,
    torch::Tensor input_x,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor dt) {
    
    // Input validation
    check_cuda(grad_output);
    check_cuda(state);
    check_cuda(input_x);
    check_cuda(A);
    check_cuda(B);
    check_cuda(dt);
    
    // These checks ensure that all input tensors in the backward pass meet the same requirements as in the forward pass. 
    // They must be on CUDA, be contiguous in memory, and have the correct dimensions.
    check_contiguous(grad_output);
    check_contiguous(state);
    check_contiguous(input_x);
    check_contiguous(A);
    check_contiguous(B);
    check_contiguous(dt);
    
    check_dim(grad_output, 3);  // [batch, d_inner, d_state]
    check_dim(state, 3);        // [batch, d_inner, d_state]
    check_dim(input_x, 2);      // [batch, d_inner]
    check_dim(A, 2);            // [d_inner, d_state]
    check_dim(B, 2);            // [d_inner, d_state]
    check_dim(dt, 2);           // [batch, d_inner]
    
    return rk4_discretize_backward_cuda(grad_output, state, input_x, A, B, dt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rk4_discretize", &rk4_discretize, "RK4 discretization (CUDA)");
    m.def("rk4_discretize_backward", &rk4_discretize_backward, "RK4 discretization backward (CUDA)");
    m.def("rk2_discretize", &rk2_discretize, "RK2 discretization (CUDA)");
    m.def("rk2_discretize_backward", &rk2_discretize_backward, "RK2 discretization backward (CUDA)");
}