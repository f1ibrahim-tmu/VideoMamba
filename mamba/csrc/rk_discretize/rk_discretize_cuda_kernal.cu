// rk_discretize_cuda_kernel.cu
#include "rk_discretize.h"
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__device__ void compute_derivative(
    const scalar_t *state,
    const scalar_t *input_x,
    const scalar_t *A,
    const scalar_t *B,
    scalar_t *output,
    const int d_state,
    const int d_inner)
{

    // Compute Ax + Bu
    for (int i = 0; i < d_inner; i++)
    {
        scalar_t sum = 0;
        for (int j = 0; j < d_state; j++)
        {
            sum += A[i * d_state + j] * state[j];
        }
        sum += B[i] * input_x[i];
        output[i] = sum;
    }
}

template <typename scalar_t>
__global__ void rk4_discretize_cuda_kernel(
    scalar_t *state,
    const scalar_t *input_x,
    const scalar_t *A,
    const scalar_t *B,
    const scalar_t *dt,
    const int batch_size,
    const int d_inner,
    const int d_state)
{

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * d_inner)
        return;

    const int batch_idx = idx / d_inner;
    const int inner_idx = idx % d_inner;

    // Allocate shared memory for intermediate results
    __shared__ scalar_t k1[1024], k2[1024], k3[1024], k4[1024];
    __shared__ scalar_t temp_state[1024];

    // RK4 steps
    compute_derivative(
        state + batch_idx * d_state,
        input_x + batch_idx * d_inner,
        A + inner_idx * d_state,
        B + inner_idx * d_inner,
        k1,
        d_state,
        d_inner);

    // k2 computation
    for (int i = 0; i < d_state; i++)
    {
        temp_state[i] = state[batch_idx * d_state + i] + 0.5f * dt[batch_idx] * k1[i];
    }
    compute_derivative(temp_state, input_x + batch_idx * d_inner, A, B, k2, d_state, d_inner);

    // k3 computation
    for (int i = 0; i < d_state; i++)
    {
        temp_state[i] = state[batch_idx * d_state + i] + 0.5f * dt[batch_idx] * k2[i];
    }
    compute_derivative(temp_state, input_x + batch_idx * d_inner, A, B, k3, d_state, d_inner);

    // k4 computation
    for (int i = 0; i < d_state; i++)
    {
        temp_state[i] = state[batch_idx * d_state + i] + dt[batch_idx] * k3[i];
    }
    compute_derivative(temp_state, input_x + batch_idx * d_inner, A, B, k4, d_state, d_inner);

    // Final update
    for (int i = 0; i < d_state; i++)
    {
        state[batch_idx * d_state + i] += (dt[batch_idx] / 6.0f) *
                                          (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
    }
}