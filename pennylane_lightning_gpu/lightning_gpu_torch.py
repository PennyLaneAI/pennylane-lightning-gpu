from .lightning_gpu import *

# Package exists only when Torch is available
try:
    import torch
    if not torch.cuda.is_available() and torch.cuda.device_count() > 0:
        raise ModuleNotFoundError("Incorrect torch version installed. Please ensure you have a CUDA 11.3+ capable version.")
    class LightningGPUTorch(LightningGPU, DefaultQubitTorch):
        def __init__(self, wires, *, shots=None, sync=True):
            super(DefaultQubitTorch, self).__init__(wires, shots=shots)
            self._state = self._state.cuda()
            self._pre_rotated_state = self._state
            self._gpu_state = _gpu_dtype(self._state.dtype, False)(self._state)
            self._sync = False

        def reset(self):
            super(DefaultQubitTorch).reset()

        def syncH2D(self, use_async=False):
            pass

        def syncD2H(self, use_async=False):
            pass
        def _apply_x(self):
            pass
        def _apply_y(self):
            pass
        def _apply_z(self):
            pass

    class RXFunc(torch.autograd.Function):
        @staticmethod
        def forward(ctx, param, state, wires):
            state_out = state.clone()
            sv = SVCudaTorch(state_out)            
            sv.apply_rx(param, wires)

            ctx.save_for_backward(params)
            return state_out

        @staticmethod
        def backward(ctx, state):
            params,jac = ctx.saved_tensors
            return grad_output * jac, None, None


except ModuleNotFoundError as e:
    class LightningGPUTorch(LightningGPU, DefaultQubitTorch):
        def __init__(self, wires, *, shots=None, sync=True):
            warn(
                "CUDA capable torch is not available. Please install and try again.",
                UserWarning,
            )
            super(LightningGPU, self).__init__(wires, *, shots=None, sync=True)