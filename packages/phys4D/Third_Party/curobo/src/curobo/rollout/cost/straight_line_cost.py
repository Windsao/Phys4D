












import torch


from curobo.util.torch_utils import get_torch_jit_decorator


from .cost_base import CostBase, CostConfig


@get_torch_jit_decorator()
def st_cost(ee_pos_batch, vec_weight, weight):
    ee_plus_one = torch.roll(ee_pos_batch, 1, dims=1)

    xdot_current = ee_pos_batch - ee_plus_one + 1e-8

    err_vec = vec_weight * xdot_current / 0.02
    error = torch.sum(torch.square(err_vec), dim=-1)


    cost = weight * error
    return cost


class StraightLineCost(CostBase):
    def __init__(self, config: CostConfig):
        CostBase.__init__(self, config)

        self.vel_idxs = torch.arange(
            self.dof, 2 * self.dof, dtype=torch.long, device=self.tensor_args.device
        )
        self.I = torch.eye(self.dof, device=self.tensor_args.device, dtype=self.tensor_args.dtype)

    def forward(self, ee_pos_batch):
        cost = st_cost(ee_pos_batch, self.vec_weight, self.weight)
        return cost
