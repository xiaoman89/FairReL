from .sampler import GroupBalancedBatchSampler
from .subgroups import SUBGROUPS, SUBGROUP_TO_ID

__all__ = ["DeepfakeDataset", "build_transform", "GroupBalancedBatchSampler",
           "SUBGROUPS", "SUBGROUP_TO_ID"]


def __getattr__(name):
    # Lazy import: DeepfakeDataset needs torchvision/PIL/pandas, which the
    # losses, metrics and unit tests do not require.
    if name in ("DeepfakeDataset", "build_transform"):
        from . import dataset
        return getattr(dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
