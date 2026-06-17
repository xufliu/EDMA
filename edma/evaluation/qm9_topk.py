from typing import Dict, Iterable, Tuple

import torch
from torch_geometric.loader import DataLoader


@torch.no_grad()
def mae_at_k_for_graph(model, z_g, pos_g, y_true_g, node_mask_g, k: int) -> Tuple[float, int]:
    _, ranked = torch.sort(node_mask_g, descending=True)
    selected = ranked[:k].sort().values

    y_sub = model(z_g[selected], pos_g[selected], batch=None).view(-1)
    y_true_g = y_true_g.view(-1)
    return torch.abs(y_sub - y_true_g).item(), int(selected.numel())


def run_qm9_topk_eval(
    explainer,
    dataset,
    model,
    target_attr: int,
    k_list: Iterable[int],
    device: torch.device,
    batch_size: int = 128,
    shuffle: bool = False,
) -> Tuple[Dict[int, list], dict]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    results = {int(k): [] for k in k_list}
    mask_stats = {
        "mean_m": [],
        "bin_frac": [],
        "num_nodes": [],
    }

    for data in loader:
        data = data.to(device)
        y_true = data.y[:, target_attr].view(-1, 1)
        explanation = explainer(data.z, data.pos, target=y_true, batch=data.batch)

        node_mask_all = explanation.node_mask
        if node_mask_all.dim() == 2:
            node_mask_all = node_mask_all.view(-1)

        with torch.no_grad():
            mask_stats["mean_m"].append(float(node_mask_all.mean().item()))
            mask_stats["bin_frac"].append(
                float(
                    ((node_mask_all < 0.05) | (node_mask_all > 0.95))
                    .float()
                    .mean()
                    .item()
                )
            )
            mask_stats["num_nodes"].append(int(node_mask_all.numel()))

        for graph_idx in range(int(data.num_graphs)):
            node_idx = (data.batch == graph_idx).nonzero(as_tuple=False).view(-1)
            for k in results:
                if node_idx.numel() < k:
                    continue
                mae_k, _ = mae_at_k_for_graph(
                    model,
                    data.z[node_idx],
                    data.pos[node_idx],
                    y_true[graph_idx],
                    node_mask_all[node_idx],
                    k,
                )
                results[k].append(mae_k)

    return results, mask_stats
