from typing import Callable, Iterable, Optional

import torch
from torch_geometric.nn import DimeNet, DimeNetPlusPlus, SchNet
from torch_geometric.nn.pool import radius_graph


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def qm9_interaction_graph(model, z: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
    if isinstance(model, SchNet):
        edge_index, _ = model.interaction_graph(pos, batch)
        return edge_index
    if isinstance(model, (DimeNet, DimeNetPlusPlus)):
        return radius_graph(
            pos,
            r=model.cutoff,
            batch=batch,
            max_num_neighbors=model.max_num_neighbors,
        )
    raise TypeError(f"Unsupported QM9 backbone: {model.__class__.__name__}")


def evaluate_qm9_energy_parameter_set(
    *,
    explainer,
    model,
    loader,
    k_list: Iterable[int],
    device: torch.device,
    prediction_fn: Optional[Callable] = None,
) -> tuple[dict, dict, list, list]:
    """Evaluate top-k QM9 explanations with the legacy EDMA output format."""
    list_of_results = {str(int(k)): [] for k in k_list}
    close_list = {str(int(k)): [] for k in k_list}
    node_ranks = []
    node_masks = []

    if prediction_fn is None:
        prediction_fn = model

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            full_prediction = prediction_fn(data.z, data.pos, data.batch).view(-1, 1)

        explanation = explainer(
            data.z,
            data.pos,
            target=full_prediction.detach(),
            batch=data.batch,
        )
        node_mask_all = explanation.node_mask.view(-1)

        for graph_id in range(int(data.num_graphs)):
            graph_nodes = (data.batch == graph_id).nonzero(as_tuple=False).view(-1)
            graph_z = data.z[graph_nodes]
            graph_pos = data.pos[graph_nodes]
            node_mask = node_mask_all[graph_nodes]

            node_masks.append(node_mask.detach().cpu().numpy().tolist())
            _, sorted_index = torch.sort(node_mask, dim=0, descending=True)
            sorted_index = sorted_index.view(-1)
            node_ranks.append(sorted_index.detach().cpu().numpy().tolist())

            edge_index = qm9_interaction_graph(model, graph_z, graph_pos)

            for k_string in list_of_results.keys():
                k = int(k_string)
                if graph_z.size(0) < k:
                    continue

                selected_index = sorted_index[:k].sort().values
                pred = prediction_fn(
                    graph_z[selected_index],
                    graph_pos[selected_index],
                ).view(-1)
                mae = (full_prediction[graph_id].view(-1) - pred).abs().item()
                list_of_results[k_string].append(mae)

                selected = set(selected_index.detach().cpu().tolist())
                closeness = 0.0
                for edge_from, edge_to in zip(edge_index[0], edge_index[1]):
                    edge_from_i = int(edge_from.detach().cpu().item())
                    edge_to_i = int(edge_to.detach().cpu().item())
                    edge_score = float(
                        (node_mask[edge_from_i] * node_mask[edge_to_i])
                        .detach()
                        .cpu()
                        .item()
                    )
                    if edge_from_i in selected and edge_to_i in selected:
                        closeness += 1.0 - edge_score
                    else:
                        closeness += edge_score
                close_list[k_string].append(closeness)

    return list_of_results, close_list, node_ranks, node_masks
