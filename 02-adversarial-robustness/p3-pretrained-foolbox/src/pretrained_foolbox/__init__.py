"""Attack a target model at INFERENCE TIME (no training of the target needed).

DEFAULT path is fully offline: a self-trained SmallCNN on synthetic 3-channel
images. OPTIONAL path attacks a torchvision ImageNet-pretrained ResNet-18.

Public API:
    set_seed, get_device         -- reproducibility helpers
    SmallCNN                     -- offline target classifier
    load_pretrained_resnet18     -- optional pretrained target (downloads weights)
    make_synthetic               -- deterministic synthetic images in [0, 1]
    train, evaluate              -- offline training + accuracy
    fgsm_perturb, pgd_perturb    -- the from-scratch L-inf attacks
    predict, true_label_confidence -- prediction + confidence helpers
"""

from .attack import (
    fgsm_perturb,
    pgd_perturb,
    predict,
    true_label_confidence,
)
from .data import make_synthetic
from .model import SmallCNN, load_pretrained_resnet18
from .train import evaluate, train
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "SmallCNN",
    "load_pretrained_resnet18",
    "make_synthetic",
    "train",
    "evaluate",
    "fgsm_perturb",
    "pgd_perturb",
    "predict",
    "true_label_confidence",
]
