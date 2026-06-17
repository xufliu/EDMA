from typing import Optional, Tuple, Type

import torch
from torch_geometric.nn import DimeNet, DimeNetPlusPlus, SchNet

from ..data.qm9 import load_qm9_dataset, resolve_qm9_path


def load_pretrained_schnet_qm9(
    data_path: Optional[str] = None,
    target_attr: int = 0,
    device: Optional[torch.device] = None,
) -> Tuple[SchNet, tuple]:
    qm9_path = resolve_qm9_path(data_path)
    dataset = load_qm9_dataset(qm9_path)
    model, splits = SchNet.from_qm9_pretrained(qm9_path, dataset, target_attr)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, splits


def load_pretrained_dimenet_qm9(
    data_path: Optional[str] = None,
    target_attr: int = 0,
    device: Optional[torch.device] = None,
    use_dimenet_plus_plus: bool = True,
    remap_targets: bool = True,
) -> Tuple[torch.nn.Module, tuple]:
    qm9_path = resolve_qm9_path(data_path)
    dataset = load_qm9_dataset(qm9_path, remap_targets=remap_targets)
    model_cls: Type[torch.nn.Module] = (
        DimeNetPlusPlus if use_dimenet_plus_plus else DimeNet
    )
    model, splits = model_cls.from_qm9_pretrained(qm9_path, dataset, target_attr)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, splits


load_pretrained_schnet = load_pretrained_schnet_qm9
load_pretrained_dimenet = load_pretrained_dimenet_qm9
