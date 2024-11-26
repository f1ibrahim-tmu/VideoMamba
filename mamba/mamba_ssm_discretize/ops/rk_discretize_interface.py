import torch
from torch.utils.cpp_extension import load

# Compile the CUDA kernels
rk_cuda = load(
    name="rk_discretize_cuda",
    sources=["rk_discretize_cuda.cpp", "rk_discretize_cuda_kernel.cu"],
    verbose=True
)

class RK4DiscreteFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, input_x, A, B, dt):
        # Save for backward
        ctx.save_for_backward(state, input_x, A, B, dt)
        
        # Call CUDA implementation
        output = rk_cuda.rk4_discretize(state, input_x, A, B, dt)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        state, input_x, A, B, dt = ctx.saved_tensors
        
        # Get gradients from CUDA implementation
        grad_state, grad_input, grad_A, grad_B, grad_dt = rk_cuda.rk4_discretize_backward(
            grad_output, state, input_x, A, B, dt
        )
        
        return grad_state, grad_input, grad_A, grad_B, grad_dt

class RK2DiscreteFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, input_x, A, B, dt):
        ctx.save_for_backward(state, input_x, A, B, dt)
        output = rk_cuda.rk2_discretize(state, input_x, A, B, dt)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        state, input_x, A, B, dt = ctx.saved_tensors
        grad_state, grad_input, grad_A, grad_B, grad_dt = rk_cuda.rk2_discretize_backward(
            grad_output, state, input_x, A, B, dt
        )
        return grad_state, grad_input, grad_A, grad_B, grad_dt

def selective_state_update_rk4(
    state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=True
):
    """
    Selective state update using RK4 discretization.
    
    Args:
        state: (batch, d_inner, d_state) - Current state
        x: (batch, d_inner) - Input
        dt: (batch, d_inner) - Time delta
        A: (d_inner, d_state) - State matrix
        B: (d_inner, d_state) - Input matrix
        C: (d_inner, d_state) - Output matrix
        D: Optional (d_inner,) - Skip connection
        z: Optional (batch, d_inner) - Update gate
        dt_bias: Optional - Bias for dt
        dt_softplus: bool - Whether to apply softplus to dt
    
    Returns:
        y: (batch, d_inner) - Output
    """
    batch, d_inner, d_state = state.shape
    
    # Apply dt preprocessing
    if dt_bias is not None:
        dt = dt + dt_bias
    if dt_softplus:
        dt = F.softplus(dt)
    
    # Apply RK4 discretization
    new_state = RK4DiscreteFunction.apply(state, x, A, B, dt)
    
    # Compute output
    y = torch.einsum("bdn,dn->bd", new_state, C)
    
    # Apply skip connection if D is provided
    if D is not None:
        y = y + D * x
    
    # Apply update gate if provided
    if z is not None:
        y = y * torch.sigmoid(z)
    
    return y

def selective_state_update_rk2(
    state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=True
):
    """
    Selective state update using RK2 discretization.
    Similar to RK4 but using 2nd order method.
    """
    batch, d_inner, d_state = state.shape
    
    if dt_bias is not None:
        dt = dt + dt_bias
    if dt_softplus:
        dt = F.softplus(dt)
    
    new_state = RK2DiscreteFunction.apply(state, x, A, B, dt)
    
    y = torch.einsum("bdn,dn->bd", new_state, C)
    
    if D is not None:
        y = y + D * x
    
    if z is not None:
        y = y * torch.sigmoid(z)
    
    return y

class StateSpaceDiscreteRK4(torch.nn.Module):
    """
    Wrapper module for RK4 discretized state space model.
    """
    def __init__(self, d_inner, d_state, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_min = dt_min
        self.dt_max = dt_max
        
        # Initialize parameters
        self.A = torch.nn.Parameter(torch.randn(d_inner, d_state) / d_state)
        self.B = torch.nn.Parameter(torch.randn(d_inner, d_state) / d_state)
        self.C = torch.nn.Parameter(torch.randn(d_inner, d_state) / d_state)
        self.D = torch.nn.Parameter(torch.randn(d_inner))
        
        # Initialize dt projection
        dt_init = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        self.dt_proj = torch.nn.Parameter(dt_init)
    
    def forward(self, x, state=None):
        batch = x.shape[0]
        
        # Initialize state if not provided
        if state is None:
            state = torch.zeros(
                batch, self.d_inner, self.d_state,
                device=x.device, dtype=x.dtype
            )
        
        # Apply selective state update
        y = selective_state_update_rk4(
            state, x, self.dt_proj, self.A, self.B, self.C, self.D,
            dt_softplus=True
        )
        
        return y, state

def test_rk4_discretization():
    """
    Test function to verify RK4 implementation.
    """
    batch_size = 2
    d_inner = 4
    d_state = 3
    
    # Create test inputs
    state = torch.randn(batch_size, d_inner, d_state, requires_grad=True)
    x = torch.randn(batch_size, d_inner, requires_grad=True)
    A = torch.randn(d_inner, d_state, requires_grad=True)
    B = torch.randn(d_inner, d_state, requires_grad=True)
    dt = torch.rand(batch_size, d_inner, requires_grad=True)
    
    # Test forward pass
    output = RK4DiscreteFunction.apply(state, x, A, B, dt)
    
    # Test backward pass
    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    
    print("Forward pass shape:", output.shape)
    print("Gradients computed successfully")
    
if __name__ == "__main__":
    test_rk4_discretization()