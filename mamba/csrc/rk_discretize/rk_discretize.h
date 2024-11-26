// rk_discretize.h
#pragma once
#include <torch/extension.h>
#include <vector>

// Forward declarations for the main functions
torch::Tensor rk4_discretize_cuda(
    torch::Tensor state,
    torch::Tensor input_x,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor dt);

std::vector<torch::Tensor> rk4_discretize_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor state,
    torch::Tensor input_x,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor dt);

// Optional: RK2 method declarations if needed
torch::Tensor rk2_discretize_cuda(
    torch::Tensor state,
    torch::Tensor input_x,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor dt);

std::vector<torch::Tensor> rk2_discretize_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor state,
    torch::Tensor input_x,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor dt);

// Utility function declarations
void check_cuda(const torch::Tensor& tensor);
void check_contiguous(const torch::Tensor& tensor);
void check_dim(const torch::Tensor& tensor, int64_t dim);