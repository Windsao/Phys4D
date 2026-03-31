










import torch
from torch.autograd import Function


from curobo.util.logger import log_warn

try:

    from curobo.curobolib import lbfgs_step_cu
except ImportError:
    log_warn("lbfgs_step_cu not found, JIT compiling...")

    from torch.utils.cpp_extension import load


    from curobo.util_file import add_cpp_path

    lbfgs_step_cu = load(
        name="lbfgs_step_cu",
        sources=add_cpp_path(
            [
                "lbfgs_step_cuda.cpp",
                "lbfgs_step_kernel.cu",
            ]
        ),
    )


class LBFGScu(Function):
    @staticmethod
    def forward(
        ctx,
        step_vec,
        rho_buffer,
        y_buffer,
        s_buffer,
        q,
        grad_q,
        x_0,
        grad_0,
        epsilon=0.1,
        stable_mode=False,
        use_shared_buffers=True,
    ):
        m, b, v_dim, _ = y_buffer.shape

        R = lbfgs_step_cu.forward(
            step_vec,
            rho_buffer,
            y_buffer,
            s_buffer,
            q,
            grad_q,
            x_0,
            grad_0,
            epsilon,
            b,
            m,
            v_dim,
            stable_mode,
            use_shared_buffers,
        )
        step_v = R[0].view(step_vec.shape)


        return step_v

    @staticmethod
    def backward(ctx, grad_output):
        return (
            None,
            None,
            None,
            None,
            None,
            None,
        )
