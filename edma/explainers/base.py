from math import sqrt
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks
from torch_geometric.explain.config import MaskType, ModelMode


class BaseEnergyInstanceExplainer(ExplainerAlgorithm):
    """Shared EDMA node-mask training logic for QM9 energy explainers."""

    coeffs = {
        "node_feat_size": 1.0,
        "node_feat_ent": 0.1,
        "hinge_w": 0.0,
        "temp": [5.0, 0.5],
        "limit_a": -0.1,
        "limit_b": 1.1,
        "epsilon": 1e-6,
        "qz_loga": -1.0,
        "mask_logit_direction": "one_minus_zero",
    }

    def __init__(
        self,
        epochs: int = 100,
        lr: float = 0.01,
        log_every: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.log_every = log_every
        self.coeffs = {**self.coeffs, **kwargs}

        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None
        self.logit_0: Optional[Parameter] = None
        self.logit_1: Optional[Parameter] = None
        self.temperature = float(self.coeffs["temp"][0])

    def supports(self) -> bool:
        return True

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(
                f"Heterogeneous graphs are not supported in {self.__class__.__name__}"
            )

        self._train(model, x, edge_index, target=target, index=index, **kwargs)

        node_mask = self.node_mask.detach().view(-1)
        edge_mask = self.edge_mask.detach() if self.edge_mask is not None else None
        self._clean_model(model)
        return Explanation(node_mask=node_mask, edge_mask=edge_mask)

    def _train(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        raise NotImplementedError

    def _initialize_masks(self, x: Tensor, edge_index: Tensor):
        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type

        if node_mask_type != MaskType.object:
            raise ValueError("EDMA explainers require node_mask_type='object'")

        device = x.device
        num_nodes = x.size(0)
        std = 0.2
        self.logit_0 = Parameter(torch.randn(num_nodes, 1, device=device) * std)
        self.logit_1 = Parameter(torch.randn(num_nodes, 1, device=device) * std)

        if edge_mask_type == MaskType.object:
            edge_std = torch.nn.init.calculate_gain("relu") * sqrt(
                2.0 / (2 * max(num_nodes, 1))
            )
            self.edge_mask = Parameter(
                torch.randn(edge_index.size(1), device=device) * edge_std
            )
        elif edge_mask_type is None:
            self.edge_mask = None
        else:
            raise ValueError("Only object edge masks or edge_mask_type=None are supported")

    def _compute_node_mask(self, temperature: float) -> Tensor:
        if self.logit_0 is None or self.logit_1 is None:
            raise RuntimeError("Masks must be initialized before training")

        direction = self.coeffs["mask_logit_direction"]
        if direction == "one_minus_zero":
            prob_1 = torch.sigmoid((self.logit_1 - self.logit_0) / temperature)
        elif direction == "zero_minus_one":
            prob_1 = torch.sigmoid((self.logit_0 - self.logit_1) / temperature)
        else:
            raise ValueError(
                "mask_logit_direction must be 'one_minus_zero' or 'zero_minus_one'"
            )

        limit_a = self.coeffs["limit_a"]
        limit_b = self.coeffs["limit_b"]
        stretched = prob_1 * (limit_b - limit_a) + limit_a
        return F.hardtanh(stretched, min_val=0.0, max_val=1.0)

    @staticmethod
    def _node_to_edge_mask(node_mask: Tensor, edge_index: Tensor) -> Tensor:
        src_attn = node_mask[edge_index[0]].view(-1)
        dst_attn = node_mask[edge_index[1]].view(-1)
        return src_attn * dst_attn

    def _loss_regression(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return F.l1_loss(y_hat, y)

    def _loss(self, y_hat: Tensor, y: Tensor, node_mask: Tensor) -> Tensor:
        if self.model_config.mode != ModelMode.regression:
            raise ValueError("EDMA explainers support regression only")

        if y.shape != y_hat.shape:
            y = y.view_as(y_hat)

        loss = self._loss_regression(y_hat, y)
        loss = loss + node_mask.mean() * self.coeffs["node_feat_size"]

        node_feat_ent = self.coeffs.get("node_feat_ent", 0.0)
        if node_feat_ent:
            loss = loss + self._reg_w(node_mask) * node_feat_ent

        hinge_w = self.coeffs.get("hinge_w", 0.0)
        if hinge_w:
            loss = loss + self.hinge_loss(node_mask).mean() * hinge_w

        return loss

    def _get_temperature(self, epoch: int) -> float:
        temp = self.coeffs["temp"]
        return float(temp[0] * pow(temp[1] / temp[0], epoch / max(1, self.epochs - 1)))

    @staticmethod
    def hinge_loss(node_mask: Tensor) -> Tensor:
        return torch.minimum(node_mask, 1.0 - node_mask)

    def cdf_qz(self, x: Tensor) -> Tensor:
        limit_a = self.coeffs["limit_a"]
        limit_b = self.coeffs["limit_b"]
        eps = self.coeffs["epsilon"]
        qz_loga = self.coeffs["qz_loga"]

        xn = (x - limit_a) / (limit_b - limit_a)
        xn = xn.clamp(min=eps, max=1.0 - eps)
        logits = torch.log(xn) - torch.log1p(-xn)
        return torch.sigmoid((logits + qz_loga) / float(self.temperature)).clamp(
            min=eps,
            max=1.0 - eps,
        )

    def _reg_w(self, node_mask: Tensor) -> Tensor:
        return torch.sum(self.cdf_qz(node_mask))

    def _clean_model(self, model: torch.nn.Module):
        clear_masks(model)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None
