from math import sqrt
from typing import Dict, List, Tuple, Optional, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import MaskType, ModelMode


@torch.no_grad()
def precompute_schnet_graph(model, pos: Tensor, batch: Tensor):
    edge_index, edge_weight = model.interaction_graph(pos, batch)
    edge_attr = model.distance_expansion(edge_weight)
    return edge_index, edge_weight, edge_attr


class EnergyInstanceExplainer(ExplainerAlgorithm):
    coeffs = {
        "node_feat_size": 1.0,
        "node_feat_ent": 0.1,
        "hinge_w": 10.0,
        "temp": [5.0, 0.5],
        "limit_a": -0.1,
        "limit_b": 1.1,
        "epsilon": 1e-6,
        "qz_loga": -1.0,
    }

    def __init__(
        self, epochs: int = 100, lr: float = 0.005, log_every: int = 25, **kwargs
    ):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.log_every = log_every
        self.coeffs.update(kwargs)

        self.node_mask = None
        self.edge_mask = None

        
        self.logit_0 = None
        self.logit_1 = None

        

    def supports(self) -> bool:
        return True

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        pos: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(pos, dict):
            raise NotImplementedError("Heterogeneous graphs are not supported yet.")

        self._train(model, x, pos, target=target, index=index, **kwargs)

        node_mask = self.node_mask.detach().view(-1)
        edge_mask = self.edge_mask.detach() if self.edge_mask is not None else None

        self._clean_model(model)
        return Explanation(node_mask=node_mask, edge_mask=edge_mask)

    def _train(
        self,
        model: torch.nn.Module,
        z: Tensor,
        pos: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        batch = kwargs.get("batch", None)
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        
        edge_index, edge_weight = model.interaction_graph(pos, batch)
        edge_attr = model.distance_expansion(edge_weight)

        self._initialize_masks(z, edge_index)

        optimizer = torch.optim.Adam([self.logit_0, self.logit_1], lr=self.lr)
        for epoch in range(self.epochs):
            self.temperature = float(self._get_temperature(epoch))
            
            p1 = ((self.logit_0 - self.logit_1)/self.temperature).sigmoid()

            a = self.coeffs["limit_a"]
            b = self.coeffs["limit_b"]
            m = p1 * (b - a) + a
            m = F.hardtanh(m, min_val=0.0, max_val=1.0)  
            self.node_mask = m

            
            src, dst = edge_index[0], edge_index[1]
            edge_mask = (m[src] * m[dst]).view(-1) 
            self.edge_mask = edge_mask

            optimizer.zero_grad()
            clear_masks(model)
            set_masks(model, edge_mask, edge_index, apply_sigmoid=False)

            # forward with mask applied (uses your CFConv patch)
            y_hat = model(
                z,
                pos,
                batch=batch,
                edge_index_override=edge_index,
                edge_weight_override=edge_weight,
                edge_attr_override=edge_attr,
            )
            y = target

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y, m)

            loss.backward()
            optimizer.step()

            if (self.log_every is not None) and (
                epoch % self.log_every == 0 or epoch == self.epochs - 1
            ):
                with torch.no_grad():
                    mean_m = float(m.mean().item())
                    frac_near_binary = float(
                        ((m < 0.05) | (m > 0.95)).float().mean().item()
                    )
                    pred_err = float(F.l1_loss(y_hat, y).item())

        clear_masks(model)

    # -------------------------
    # init + loss + helpers
    # -------------------------
    def _initialize_masks(self, z: Tensor, edge_index: Tensor):
        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type is None:
            raise ValueError("This explainer requires node_mask_type != None")

        device = z.device
        N = z.size(0)

        std = 0.1
        if node_mask_type == MaskType.object:
            self.logit_0 = Parameter(torch.randn(N, 1, device=device) * std)
            self.logit_1 = Parameter(torch.randn(N, 1, device=device) * std)
        else:
            raise ValueError("Only MaskType.object is supported for this explainer")

    def _loss_regression(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return F.l1_loss(y_hat, y)

    def _loss(self, y_hat: Tensor, y: Tensor, node_mask: Tensor) -> Tensor:
        # base loss
        if self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)
        else:
            raise ValueError("Only regression is supported in your current setup")

        # regularizers (your style, but now consistent)
        loss = loss + node_mask.mean() * self.coeffs["node_feat_size"]

        return loss

    def _get_temperature(self, epoch: int) -> float:
        t0, t1 = self.coeffs["temp"]
        return t0 * pow(t1 / t0, epoch / max(1, self.epochs - 1))

    def cdf_qz(self, x: Tensor) -> Tensor:
        a = self.coeffs["limit_a"]
        b = self.coeffs["limit_b"]
        eps = self.coeffs["epsilon"]
        qz_loga = self.coeffs["qz_loga"]

        # IMPORTANT: x is in [0,1]; map to stretched domain for the cdf computation
        xn = (x - a) / (b - a)
        xn = xn.clamp(min=eps, max=1.0 - eps)
        logits = torch.log(xn) - torch.log(1.0 - xn)

        return torch.sigmoid((logits + qz_loga) / float(self.temperature)).clamp(
            min=eps, max=1.0 - eps
        )

    def _reg_w(self, node_mask: Tensor) -> Tensor:
        
        return torch.sum(self.cdf_qz(node_mask))

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = None
        self.edge_mask = None
