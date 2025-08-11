import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import copy
import glob
from dataclasses import dataclass
from functools import lru_cache, partial # Added partial for hook registration
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
import math
import tiktoken
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not available. Install with \'pip install wandb\' for experiment tracking.")
    WANDB_AVAILABLE = False
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention
#torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)

# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        # Efficient systems-wise implementation of step developed by @YouJiacheng,
        # @KonstantinWilleke, @alexrgilbert, @adricarda, @tuttyfrutyee, @vdlad,
        # @ryanyang0, and @vagrawal.
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            grad_pad = [param.grad for param in params] + [torch.zeros_like(params[-1])] * world_size
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    grad = params[base_i + rank].grad
                # This gives strange dynamo warnings
                reduce_scatter_futures.append(dist.reduce_scatter(grad, grad_pad[base_i:base_i + world_size], op=dist.ReduceOp.AVG, async_op=True).get_future())

        idx = 0
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * world_size
            momentum = group["momentum"]
            for base_i in range(0, len(params), world_size):
                reduce_scatter_futures[idx].wait()
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    grad = p.grad
                    eff_lr = group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5 * getattr(p, "lr_mul", 1.0)
                    eff_weight_decay = group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    momentum_buffer = state["momentum_buffer"]
                    p.mul_(1 - eff_weight_decay)
                    momentum_buffer.lerp_(grad, 1 - momentum)
                    grad = grad.lerp_(momentum_buffer, momentum)
                    v = zeropower_via_newtonschulz5(grad.bfloat16(), 5)
                    p.add_(other=v, alpha=-eff_lr)
                idx += 1
                all_reduce_futures.append(dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank], async_op=True).get_future())
        torch.futures.collect_all(all_reduce_futures).wait()

class DistAdam(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)
        # DistributedAdam implementation by @vagrawal

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            for base_i in range(len(params)):
                grad = params[base_i].grad
                if grad is None:
                    print(f"ERROR: Parameter {base_i} in group has None gradient!")
                    print(f"Parameter shape: {params[base_i].shape}")
                    print(f"Parameter name: {[name for name, param in model.named_parameters() if param is params[base_i]]}")
                    print(f"Parameter requires_grad: {params[base_i].requires_grad}")
                    # Create zero gradient to prevent crash
                    raise ValueError(f"Parameter {base_i} in group has None gradient!")
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for base in range(len(params)):
                reduce_scatter_futures[idx].wait()
                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice = grad_slices[idx]
                # State init
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                # bias corrections
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                idx += 1
                all_reduce_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
        torch.futures.collect_all(all_reduce_futures).wait()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.12

    def forward(self, x: Tensor, ve: Tensor | None, lambdas: Tensor, block_mask: BlockMask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = lambdas[0] * v + lambdas[1] * ve.view_as(v) # @KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = lambdas[0] * v
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=self.attn_scale).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

# -----------------------------------------------------------------------------
# Mamba2 Integration

try:
    from mamba_ssm.modules.mamba2 import Mamba2
    # Also check if causal_conv1d is available
    try:
        from causal_conv1d.cpp_functions import causal_conv1d_fwd_function
    except ImportError:
        raise ImportError("causal_conv1d not properly installed. Mamba layers may not work correctly in distributed training.")
        
    MAMBA_AVAILABLE = True
except ImportError:
    print("Warning: mamba_ssm with Mamba2 not found. Mamba layers will be disabled.")
    MAMBA_AVAILABLE = False
    Mamba2 = None

class MambaBlock(nn.Module):
    """Mamba2 block with proper hybrid tensor/data parallelism"""
    def __init__(self, dim: int, layer_idx: int = None):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm with Mamba2 is required for Mamba layers")
        
        # Hybrid Tensor/Data Parallelism Solution:
        # - Enable Mamba tensor parallelism for memory efficiency  
        # - Disable problematic features that cause stride/tensor size issues
        # - Let DDP wrap everything for gradient synchronization
        
        world_size = 1
        process_group = None
        # if dist.is_initialized():
        #     world_size = dist.get_world_size()
        #     process_group = dist.group.WORLD
        
        self.world_size = world_size
        
        self.mamba = Mamba2(
            d_model=dim,
            d_state=64,     # Reduced from 128 to save memory
            d_conv=4,
            expand=2,       # Keep standard expand
            ngroups=1,      # Keep simple since process_group=None
            layer_idx=layer_idx,
            process_group=process_group,  # None for single-GPU-mode per layer
            sequence_parallel=False,      # Disabled for compatibility
            use_mem_eff_path=False,       # Disabled to avoid causal_conv1d issues
            device=None,
            dtype=None
        )
        
        # Don't override activation - that causes assertion errors
        # Instead, we'll handle the stride issue differently in the forward pass
    
    def forward(self, x: Tensor, ve: Tensor | None = None, lambdas: Tensor | None = None, block_mask=None):
        """
        Forward pass compatible with attention layer interface.
        Args:
            x: Input tensor (B, L, D)
            ve: Value embeddings (ignored for Mamba)
            lambdas: Lambda weights (ignored for Mamba)
            block_mask: Block mask (ignored for Mamba)
        Returns:
            Output tensor (B, L, D)
        """
        # Ensure input matches expected dtype for distributed operations
        x_input = x.to(dtype=torch.bfloat16) if x.dtype != torch.bfloat16 else x
        
        # Ensure tensor is contiguous to avoid stride issues in causal_conv1d
        x_input = x_input.contiguous()
        
        # Advanced approach: Temporarily disable causal_conv1d to force fallback path
        # This avoids stride issues while keeping proper activation
        
        # Import the module where causal_conv1d_fn is used
        import mamba_ssm.modules.mamba2 as mamba2_module
        
        # Save original causal_conv1d_fn
        original_causal_conv1d_fn = getattr(mamba2_module, 'causal_conv1d_fn', None)
        
        try:
            # Temporarily set causal_conv1d_fn to None to force fallback path
            mamba2_module.causal_conv1d_fn = None
            
            # Now run Mamba forward - it will use the fallback conv1d path
            output = self.mamba(x_input)
            
        finally:
            # Restore original causal_conv1d_fn
            mamba2_module.causal_conv1d_fn = original_causal_conv1d_fn
        
        return output

class Block(nn.Module):
    """Block supporting both Attention and Mamba layers based on architecture string"""
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int, arch: str = 'T'):
        super().__init__()
        
        # Architecture-based layer selection
        if arch.lower() == 'm' and MAMBA_AVAILABLE:
            # Mamba layer
            self.mixer = MambaBlock(dim, layer_idx=layer_idx)
            self.mixer_type = "mamba"
        elif arch.lower() == 'n':
            # No mixer (skip layer) - only if explicitly specified in architecture
            self.mixer = None
            self.mixer_type = "none"
        elif layer_idx == 7 and arch.lower() == 't':
            # Skip attention of blocks.7 (the 8th layer) by @YouJiacheng - only for standard attention
            self.mixer = None
            self.mixer_type = "none"
        else:
            # Standard attention layer (default)
            self.mixer = CausalSelfAttention(dim, num_heads, max_seq_len)
            self.mixer_type = "attention"
            
        # Add MLP for uppercase architectures (T, M) or if not specified
        if arch.isupper() or len(arch) == 0:
            self.mlp = MLP(dim)
        else:
            self.mlp = nn.Identity()

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, lambdas: Tensor, sa_lambdas: Tensor, block_mask: BlockMask):
        # Simple forward pass - no gradient checkpointing to avoid DDP issues
        x = lambdas[0] * x + lambdas[1] * x0
        if self.mixer is not None:
            if self.mixer_type == "mamba":
                # Mamba doesn't use value embeddings, block_mask, or sa_lambdas  
                x = x + self.mixer(norm(x), None, None, None)
            elif self.mixer_type == "attention":
                # Standard attention path
                x = x + self.mixer(norm(x), ve, sa_lambdas, block_mask)
        x = x + self.mlp(norm(x))
        return x

def parse_architecture_string(arch_str: str, num_layers: int):
    """
    Parse architecture string into list of layer types.
    
    Args:
        arch_str: Architecture string like "tmtmtm" or "TMmMtT"
        num_layers: Total number of layers
        
    Returns:
        List of architecture characters, repeated/truncated to match num_layers
    """
    if not arch_str:
        # Default to all attention with MLP
        return ['T'] * num_layers
    
    # If string is shorter than num_layers, repeat it
    if len(arch_str) < num_layers:
        repeats = (num_layers + len(arch_str) - 1) // len(arch_str)
        arch_str = arch_str * repeats
    
    # Truncate to exact length
    return list(arch_str[:num_layers])

def create_architecture_pattern(pattern_name: str, num_layers: int):
    """
    Create common architecture patterns.
    
    Args:
        pattern_name: Name of the pattern:
                     - "pure_attention": All attention layers (T)
                     - "pure_mamba": All Mamba layers (M)  
                     - "alternating": Alternating attention-mamba (tmtmtm...)
                     - "alternating_blocks": Alternating blocks of 2 (ttmmttmm...)
                     - "sandwich": Attention-mamba-attention sandwich (T...M...T)
        num_layers: Total number of layers
        
    Returns:
        Architecture string
    """
    if pattern_name == "pure_attention":
        return "T" * num_layers
    elif pattern_name == "pure_mamba":
        return "M" * num_layers
    elif pattern_name == "alternating":
        return "".join(["t" if i % 2 == 0 else "m" for i in range(num_layers)])
    elif pattern_name == "alternating_blocks":
        pattern = ""
        for i in range(num_layers):
            if (i // 2) % 2 == 0:
                pattern += "t"
            else:
                pattern += "m"
        return pattern
    elif pattern_name == "sandwich":
        if num_layers < 3:
            return "T" * num_layers
        # First 1/3 attention, middle 1/3 mamba, last 1/3 attention
        third = num_layers // 3
        remainder = num_layers % 3
        return "T" * (third + remainder) + "M" * third + "T" * third
    else:
        raise ValueError(f"Unknown pattern: {pattern_name}")

# Convenience functions for common patterns
def make_pure_attention(num_layers: int) -> str:
    """Create architecture string for pure attention model."""
    return create_architecture_pattern("pure_attention", num_layers)

def make_pure_mamba(num_layers: int) -> str:
    """Create architecture string for pure Mamba model.""" 
    return create_architecture_pattern("pure_mamba", num_layers)

def make_alternating(num_layers: int) -> str:
    """Create architecture string for alternating attention-mamba."""
    return create_architecture_pattern("alternating", num_layers)



# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int, architecture: str = None):
        """
        GPT model with optional Mamba layer support.
        
        Args:
            architecture: Optional architecture string. Each character specifies a layer type:
                         - 'T': Attention with MLP (default)
                         - 't': Attention without MLP  
                         - 'M': Mamba with MLP
                         - 'm': Mamba without MLP
                         If None, defaults to all attention with MLP.
        """
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        
        # Parse architecture string or default to all attention
        if architecture is None:
            arch_layers = ['T'] * num_layers  # Default: all attention with MLP
        else:
            arch_layers = parse_architecture_string(architecture, num_layers)
        
        # Create blocks - minimal change from train_gpt.py
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, max_seq_len, i, arch_layers[i]) 
            for i in range(num_layers)
        ])
        
        # Store architecture for logging
        self.architecture = ''.join(arch_layers) if architecture else 'T' * num_layers
        
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(model_dim, vocab_size, use_fp8=True, x_s=(model_dim**0.5)/448, w_s=24/448, grad_s=1/448)
        self.lm_head.weight.detach().zero_() # @Grad62304977
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        pad = (-num_layers * 5) % dist.get_world_size()
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers), # skip_weights
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)], # block lambdas
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)], # SA lambdas
            torch.ones(pad),
        ]))
        # set learning rates
        for param in self.embed.parameters():
            param.lr_mul = 75.
        for param in self.value_embeds.parameters():
            param.lr_mul = 75.
        self.lm_head.weight.lr_mul = 27.5
        self.scalars.lr_mul = 5.0

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        # manual block mask creation by @YouJiacheng
        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)
        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )
        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor, return_logits=False):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
        assert len(block_masks) == len(self.blocks)

        x = x0 = norm(self.embed(input_seq)[None]) # use of norm here by @Grad62304977

        # U-net design by @brendanh0gan
        skip_connections = []
        skip_weights = self.scalars[:(len(self.blocks) // 2)]
        lambdas = self.scalars[1 * len(self.blocks): 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks): 5 * len(self.blocks)].view(-1, 2)

        n = len(self.blocks) // 2

        for i in range(len(self.blocks)):
            if i >= n:
                x = x + skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        logits = self.lm_head(x).float()
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
        
        if return_logits:
            return logits
            
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, reduction="sum" if self.training else "mean")
        return loss

# -----------------------------------------------------------------------------
# Distributed data loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

# find world_size starting indicies, such that each begins with token 50256 and local_batches don't overlap
def find_batch_starts(tokens: Tensor, pos: int, local_batch_size: int, max_batch_span: int):
    boundary_mask = tokens[pos : pos + max_batch_span] == 50256
    boundary_positions = torch.nonzero(boundary_mask, as_tuple=False).squeeze(-1) + pos
    start = boundary_positions[0].item()
    starts = []
    for i in range(1, len(boundary_positions)):
        end = boundary_positions[i].item() 
        if end - start >= local_batch_size:
            starts.append(start) # append start once end pos is confirmed
            if len(starts) == dist.get_world_size():
                return starts, end - pos
            start = end
    assert False # increase max_batch_span if necessary

def distributed_data_generator(filename_pattern: str, batch_size: int, align_to_bos: bool):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    max_batch_span = 2 * batch_size if align_to_bos else batch_size # provide buffer to handle samples up to length local_batch_size
    while True:
        if pos + max_batch_span + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        if align_to_bos:
            batch_starts, batch_span = find_batch_starts(tokens, pos, local_batch_size, max_batch_span)
            start_idx = batch_starts[rank]
        else:
            batch_span = batch_size
            start_idx = pos + rank * local_batch_size
        buf = tokens[start_idx:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
        pos += batch_span
        yield inputs, targets

# -----------------------------------------------------------------------------
# Text generation for evaluation

def generate_text_sample_gpt(model, tokenizer, device, max_tokens=100, temperature=0.8):
    """Generate a sample text to evaluate model quality"""
    model.eval()
    
    # Start with a simple prompt
    prompt = "The quick brown fox"
    prompt_tokens = tokenizer.encode(prompt)
    
    # Convert to tensor
    tokens = torch.tensor(prompt_tokens, dtype=torch.int32, device=device)
    generated = tokens.tolist()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get model logits (need to pad to block size)
            seq_len = len(tokens)
            if seq_len % 128 != 0:
                pad_len = 128 - (seq_len % 128)
                tokens_padded = F.pad(tokens, (0, pad_len), value=50256)  # Pad with EOS token
            else:
                tokens_padded = tokens
                
            # Create dummy targets and window size for generation
            dummy_targets = torch.zeros_like(tokens_padded[:-1])
            window_size = torch.tensor(max(1, len(tokens_padded) // 128), dtype=torch.int32, device=device)
            
            # Forward pass to get logits
            logits = model(tokens_padded, dummy_targets, window_size, return_logits=True)
            
            # Get logits for the last real token (not padding)
            last_logits = logits[0, len(tokens) - 1, :] / temperature
            
            # Sample next token
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            if next_token == 50256:  # Stop at EOS (GPT-2 endoftext token)
                break
                
            generated.append(next_token)
            tokens = torch.tensor(generated, dtype=torch.int32, device=device)
            
            # Stop if we've generated enough
            if len(generated) - len(prompt_tokens) >= max_tokens:
                break
    
    # Decode back to text
    try:
        return tokenizer.decode(generated)
    except:
        return f"[Failed to decode: {generated}]"

def calculate_metrics_gpt(loss, input_tokens, target_tokens, tokenizer):
    """Calculate perplexity and bits per byte for GPT with BPE tokenization"""
    perplexity = torch.exp(loss).item()
    
    # For BPE tokenization, we need to calculate bits per byte
    # by converting tokens back to text and measuring byte length
    try:
        # Convert a sample of tokens to text to estimate compression ratio
        sample_tokens = target_tokens[:min(1000, len(target_tokens))].cpu().tolist()
        sample_text = tokenizer.decode(sample_tokens)
        sample_bytes = len(sample_text.encode('utf-8'))
        
        # Estimate bits per byte
        # loss is nats per token, convert to bits per byte
        bits_per_token = loss.item() / math.log(2)  # Convert nats to bits
        tokens_per_byte = len(sample_tokens) / sample_bytes if sample_bytes > 0 else 1.0
        bits_per_byte = bits_per_token * tokens_per_byte
        
    except:
        bits_per_byte = float('nan')  # Fallback if decoding fails
    
    return {
        'perplexity': perplexity,
        'bits_per_byte': bits_per_byte,
        'loss': loss.item()
    }

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_seq_len = 48*1024 # FlexAttention sequence length
    val_seq_len = 4*64*1024 # FlexAttention sequence length for validation
    # optimization
    num_iterations = 1750 + 750 # number of iterations to run
    cooldown_frac = 0.45 # fraction of training spent cooling down the learning rate
    # evaluation and logging
    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint = False

    # Model architecture - can be overridden via command line
    vocab_size = 50257
    num_layers = 12
    num_heads = 6
    model_dim = 768
    # architecture = "tmtmtmtmtmtm"  # Default hybrid pattern: alternating attention-mamba
    architecture = "pure_attention"
    
    # Experiment settings
    experiment_name = "default"
    output_dir = "logs"  # Output directory for logs

# Command line argument parsing for experiments
def parse_experiment_args():
    """Parse command line arguments for experiment configurations"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Mamba2-GPT Hybrid Model")
    parser.add_argument("--experiment-name", type=str, default="default", help="Name of experiment")
    parser.add_argument("--model-dim", type=int, default=768, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--architecture", type=str, default="pure_mamba", 
                        help="Architecture string using hnet notation. Each char specifies layer type: "
                             "t=attention, m=mamba, T=attention+MLP, M=mamba+MLP, n=no mixer. "
                             "Examples: 'tmtm', 'TMTM', 'ttmmttmm'. "
                             "Also supports patterns: 'pure_attention', 'pure_mamba', 'alternating', "
                             "'alternating_blocks', 'sandwich'")

    parser.add_argument("--num-iterations", type=int, default=None, help="Override number of training iterations")
    parser.add_argument("--output-dir", type=str, default="logs", help="Output directory for logs")
    
    args = parser.parse_args()
    return args

def update_args_from_cmdline(args: Hyperparameters):
    """Update hyperparameters from command line arguments"""
    cmd_args = parse_experiment_args()
    
    args.experiment_name = cmd_args.experiment_name
    args.model_dim = cmd_args.model_dim
    args.num_layers = cmd_args.num_layers
    args.num_heads = cmd_args.num_heads
    args.output_dir = cmd_args.output_dir
    
    # Set architecture string - support both patterns and direct strings
    arch = cmd_args.architecture
    if arch in ["pure_attention", "pure_mamba", "alternating", "alternating_blocks", "sandwich"]:
        args.architecture = create_architecture_pattern(arch, args.num_layers)
    else:
        args.architecture = arch
    
    # Override iterations if specified
    if cmd_args.num_iterations is not None:
        args.num_iterations = cmd_args.num_iterations
    
    return args

args = Hyperparameters()

# Check if we're being called from command line (for experiments)
if len(sys.argv) > 1:
    args = update_args_from_cmdline(args)

# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert world_size == 4 # TODO: we are testing with 4xH100, but this code is designed for 8xH100
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.experiment_name}_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)
    logfile = f"{args.output_dir}/{run_id}.txt"
    print(logfile)
    
    # Initialize wandb if available (only on master process)
    if WANDB_AVAILABLE:
        wandb.init(
            project="speedrun",
            entity="speedrun",
            name=run_id,
            config={
                "experiment_name": args.experiment_name,
                "model_dim": args.model_dim,
                "num_layers": args.num_layers, 
                "num_heads": args.num_heads,
                "architecture": args.architecture,
                "num_iterations": args.num_iterations,
                "train_seq_len": args.train_seq_len,
                "val_seq_len": args.val_seq_len,
                "cooldown_frac": args.cooldown_frac,
                "vocab_size": args.vocab_size,
            },
            save_code=True
        )
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)

model: nn.Module = GPT(vocab_size=args.vocab_size, num_layers=args.num_layers, num_heads=args.num_heads, model_dim=args.model_dim, 
                       max_seq_len=max(args.train_seq_len, args.val_seq_len),
                       architecture=args.architecture).cuda()

# DDP setup with unused parameter detection
# Some layers may not be used (e.g., skipped attention in layer 7)
# or parameters might not receive gradients in mixed architectures
# from torch.nn.parallel import DistributedDataParallel as DDP

# # Use find_unused_parameters=True to handle unused parameters gracefully
# model = DDP(model, find_unused_parameters=True)

# Log model architecture info (access underlying model if wrapped in DDP)
underlying_model = model.module if hasattr(model, 'module') else model
if hasattr(underlying_model, 'architecture') and underlying_model.architecture != 'T' * args.num_layers:
    print0(f"Experiment: {args.experiment_name}")
    print0(f"Model Architecture: {underlying_model.architecture}")
    attention_count = underlying_model.architecture.count('t') + underlying_model.architecture.count('T')
    mamba_count = underlying_model.architecture.count('m') + underlying_model.architecture.count('M') 
    print0(f"Architecture: {attention_count} attention layers, {mamba_count} mamba layers")
else:
    print0(f"Experiment: {args.experiment_name} (standard GPT architecture)")

print0(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Convert embeddings to bfloat16 (same as train_gpt.py)
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()

# Additional fix: Ensure Mamba2 modules use consistent dtypes
# Note: We keep A_log in float32 for numerical stability (see mamba2.py:181 comment)
# but ensure other parameters and linear layers match the model dtype
for m in model.modules():
    if hasattr(m, 'mamba'):
        mamba = m.mamba
        # Convert parameters that can safely use bfloat16
        if hasattr(mamba, 'dt_bias'):
            mamba.dt_bias.data = mamba.dt_bias.data.to(torch.bfloat16)
        if hasattr(mamba, 'D'):
            mamba.D.data = mamba.D.data.to(torch.bfloat16)
        
        # Ensure linear layers use bfloat16 for weights and biases
        for name, param in mamba.named_parameters():
            if 'A_log' not in name and param.dtype == torch.float32:
                # Convert all non-A_log parameters to bfloat16
                param.data = param.data.to(torch.bfloat16)
        
        # Specifically ensure distributed linear layers have correct dtype
        if hasattr(mamba, 'in_proj') and hasattr(mamba.in_proj, 'weight'):
            mamba.in_proj.weight.data = mamba.in_proj.weight.data.to(torch.bfloat16)
            if mamba.in_proj.bias is not None:
                mamba.in_proj.bias.data = mamba.in_proj.bias.data.to(torch.bfloat16)
        if hasattr(mamba, 'out_proj') and hasattr(mamba.out_proj, 'weight'):
            mamba.out_proj.weight.data = mamba.out_proj.weight.data.to(torch.bfloat16)
            if mamba.out_proj.bias is not None:
                mamba.out_proj.bias.data = mamba.out_proj.bias.data.to(torch.bfloat16)

for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
# Need to be careful with DDP wrapped model
underlying_model = model.module if hasattr(model, 'module') else model

hidden_matrix_params = [p for n, p in underlying_model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
embed_params = [p for n, p in underlying_model.named_parameters() if "embed" in n]
scalar_params = [p for p in underlying_model.parameters() if p.ndim < 2]
head_params = [underlying_model.lm_head.weight]

# Debug parameter collection
print0(f"Collected {len(hidden_matrix_params)} hidden matrix params")
print0(f"Collected {len(embed_params)} embedding params")
print0(f"Collected {len(scalar_params)} scalar params")
print0(f"Collected {len(head_params)} head params")

# Check for any parameters that might be missing
all_collected_params = set(hidden_matrix_params + embed_params + scalar_params + head_params)
all_model_params = set(model.parameters())
missing_params = all_model_params - all_collected_params

if missing_params:
    print0(f"WARNING: {len(missing_params)} parameters not collected for optimization!")
    for i, param in enumerate(missing_params):
        param_names = [name for name, p in model.named_parameters() if p is param]
        print0(f"  Missing param {i}: shape={param.shape}, names={param_names}")

# init the optimizer(s)
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer1 = DistAdam(scalar_params + head_params + embed_params, lr=0.008, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0)
optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, weight_decay=0.0)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay
def get_lr(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / args.cooldown_frac
        return w * 1.0 + (1 - w) * 0.1

# attention window size schedule: linearly increase
@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
def get_window_size_blocks(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)

model: nn.Module = torch.compile(model, dynamic=False)

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 10
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, align_to_bos=True)
for step_i in range(warmup_steps):
    print(f"Warmup step {step_i}")
    inputs, targets = next(train_loader)
    model(inputs, targets, get_window_size_blocks(1)).backward()
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del train_loader, initial_state

########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, align_to_bos=True)
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_batch_size = world_size * args.val_seq_len
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(args.val_files, val_batch_size, align_to_bos=False)
        val_loss = 0
        sample_inputs, sample_targets = None, None
        
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                if i >= val_steps:
                    break
                val_loss += model(inputs, targets, get_window_size_blocks(step))
                
                # Save sample for metrics calculation
                if i == 0:
                    sample_inputs, sample_targets = inputs, targets
        
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        
        # Calculate metrics
        tokenizer = tiktoken.get_encoding("gpt2")
        metrics = calculate_metrics_gpt(val_loss, sample_inputs, sample_targets, tokenizer)
        
        # Generate text sample (only on master process to avoid spam)
        sample_text = ""
        if master_process and step % (args.val_loss_every * 2) == 0:  # Generate less frequently
            try:
                sample_text = generate_text_sample_gpt(model, tokenizer, device, max_tokens=50, temperature=0.8)
                print0(f"Generated sample: '{sample_text}'", console=True)
            except Exception as e:
                print0(f"Generation failed: {e}", console=True)
        
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} perplexity:{metrics['perplexity']:.2f} bits_per_byte:{metrics['bits_per_byte']:.3f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        
        # Log to wandb if available (only on master process)
        if master_process and WANDB_AVAILABLE:
            wandb.log({
                "val/loss": val_loss.item(),
                "val/perplexity": metrics['perplexity'],
                "val/bits_per_byte": metrics['bits_per_byte'],
                "train/step": step,
                "train/training_time_ms": training_time_ms,
                "train/step_avg_ms": training_time_ms/max(step, 1),
                "train/tokens_processed": step * world_size * args.train_seq_len,
                "train/window_size_blocks": get_window_size_blocks(step).item(),
            }, step=step)
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"{args.output_dir}/{run_id}", exist_ok=True)
            torch.save(log, f"{args.output_dir}/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    model(inputs, targets, get_window_size_blocks(step)).backward()
    # set optimization hyperparameters
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1) # momentum warmup for muon
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers
    for opt in optimizers:
        opt.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
# Finish wandb run
if master_process and WANDB_AVAILABLE:
    wandb.finish()

dist.destroy_process_group()