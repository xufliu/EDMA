from typing import Optional, Union

import torch
from torch import Tensor

from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.nn import DimeNet, DimeNetPlusPlus
from torch_geometric.nn.models.dimenet import triplets
from torch_geometric.nn.pool import radius_graph
from torch_geometric.utils import scatter

from .base import BaseEnergyInstanceExplainer


class DimeNetEnergyInstanceExplainer(BaseEnergyInstanceExplainer):
    """EDMA instance explainer for PyG DimeNet and DimeNet++ QM9 models."""

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
        z = x
        pos = edge_index
        batch = kwargs.get("batch")
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        edge_index = radius_graph(
            pos,
            r=model.cutoff,
            batch=batch,
            max_num_neighbors=model.max_num_neighbors,
        )
        self._initialize_masks(z, edge_index)

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

                y_hat = self._masked_forward(model, z, pos, batch, edge_index)
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
    ) -> Tensor:
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index,
            num_nodes=z.size(0),
        )
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        if isinstance(model, DimeNetPlusPlus):
            pos_jk = pos[idx_j] - pos[idx_k]
            pos_ij = pos[idx_i] - pos[idx_j]
            a = (pos_ij * pos_jk).sum(dim=-1)
            b = torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1)
        elif isinstance(model, DimeNet):
            pos_ji = pos[idx_j] - pos[idx_i]
            pos_ki = pos[idx_k] - pos[idx_i]
            a = (pos_ji * pos_ki).sum(dim=-1)
            b = torch.cross(pos_ji, pos_ki, dim=1).norm(dim=-1)
        else:
            raise TypeError("DimeNetEnergyInstanceExplainer requires DimeNet/DimeNet++")

        angle = torch.atan2(b, a)
        rbf = model.rbf(dist)
        sbf = model.sbf(dist, angle, idx_kj)

        x = model.emb(z, rbf, i, j)
        x = x * self.edge_mask.unsqueeze(1)
        p = model.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        for interaction_block, output_block in zip(
            model.interaction_blocks,
            model.output_blocks[1:],
        ):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            p = p + output_block(x, rbf, i, num_nodes=pos.size(0))

        return scatter(p, batch, dim=0, reduce="sum")


EnergyInstanceExplainer = DimeNetEnergyInstanceExplainer
