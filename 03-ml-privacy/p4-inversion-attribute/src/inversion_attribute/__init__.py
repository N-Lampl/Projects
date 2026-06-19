"""Model inversion + attribute inference: two ML privacy attacks from scratch.

Public API:
    set_seed, get_device        -- reproducibility helpers
    SmallCNN                    -- the target image classifier
    get_loaders, make_synthetic, class_prototypes  -- offline image data
    train, evaluate             -- train/eval the target
    invert_class, invert_all_classes, reconstruction_quality  -- gradient-ascent inversion
    make_attribute_dataset, train_target,
    run_attribute_inference     -- sklearn attribute-inference attack
"""

from .attribute import (
    make_attribute_dataset,
    run_attribute_inference,
    train_target,
)
from .data import class_prototypes, get_loaders, make_synthetic
from .inversion import (
    invert_all_classes,
    invert_class,
    reconstruction_quality,
)
from .model import SmallCNN
from .train import evaluate, train
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "SmallCNN",
    "get_loaders",
    "make_synthetic",
    "class_prototypes",
    "train",
    "evaluate",
    "invert_class",
    "invert_all_classes",
    "reconstruction_quality",
    "make_attribute_dataset",
    "train_target",
    "run_attribute_inference",
]
