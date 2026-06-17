from typing import Optional, Union

import torch
from torch import Tensor

from torch_geometric.explain.algorithm.utils import clear_masks, set_masks

from .base import BaseEnergyInstanceExplainer


class SchNetEnergyInstanceExplainer(BaseEnergyInstanceExplainer):
    """EDMA instance explainer for PyG SchNet QM9 models."""

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
        pos = edge_index
        batch = kwargs.get("batch")
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        edge_index, edge_weight = model.interaction_graph(pos, batch)
        self._initialize_masks(x, edge_index)

        optimizer = torch.optim.Adam([self.logit_0, self.logit_1], lr=self.lr)

        training = model.training
        requires_grad = [p.requires_grad for p in model.parameters()]
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        try:
            for epoch in range(self.epochs):
                self.temperature = self._get_temperature(epoch)
                self.node_mask = self._compute_node_mask(self.temperature)
                self.edge_mask = self._node_to_edge_mask(self.node_mask, edge_index)

                optimizer.zero_grad()
                clear_masks(model)
                set_masks(model, self.edge_mask, edge_index, apply_sigmoid=False)

                y_hat = self._masked_forward(model, x, pos, batch, edge_index, edge_weight)
                y = target

                if index is not None:
                    y_hat, y = y_hat[index], y[index]

                loss = self._loss(y_hat, y, self.node_mask)
                loss.backward()
                optimizer.step()
        finally:
            clear_masks(model)
            model.train(training)
            for parameter, requires_grad_value in zip(model.parameters(), requires_grad):
                parameter.requires_grad_(requires_grad_value)

    def _masked_forward(
        self,
        model: torch.nn.Module,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        h = model.embedding(z)
        edge_attr = model.distance_expansion(edge_weight)

        for interaction in model.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = model.lin1(h)
        h = model.act(h)
        h = model.lin2(h)

        if model.dipole:
            mass = model.atomic_mass[z].view(-1, 1)
            mass_sum = model.sum_aggr(mass, batch, dim=0)
            center = model.sum_aggr(mass * pos, batch, dim=0) / mass_sum
            h = h * (pos - center.index_select(0, batch))

        if not model.dipole and model.mean is not None and model.std is not None:
            h = h * model.std + model.mean

        if not model.dipole and model.atomref is not None:
            h = h + model.atomref(z) * self.node_mask

        out = model.readout(h, batch, dim=0)

        if model.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if model.scale is not None:
            out = model.scale * out

        return out


EnergyInstanceExplainer = SchNetEnergyInstanceExplainer
