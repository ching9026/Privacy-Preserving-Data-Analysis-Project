import os
import random

import numpy as np
import torch

from config import base_options
from trainers.pFedCTP_ES import pFedCTP_ES


def upsert_argument(parser, *name_or_flags, **kwargs):
    """
    Add an argument if it does not exist.
    If it already exists in config.py, update its type/default/choices/help.

    This allows run_pFedCTP_ES.py to work with:
        original config.py
        friend's config.py
    without directly modifying config.py.
    """

    target_flags = set(name_or_flags)

    for action in parser._actions:
        if target_flags.intersection(set(action.option_strings)):
            if "default" in kwargs:
                action.default = kwargs["default"]
            if "type" in kwargs:
                action.type = kwargs["type"]
            if "choices" in kwargs:
                action.choices = kwargs["choices"]
            if "help" in kwargs:
                action.help = kwargs["help"]
            return

    parser.add_argument(*name_or_flags, **kwargs)


def read_options():
    parser = base_options()

    # ============================================================
    # Fine-tuning mode
    # ============================================================

    upsert_argument(
        parser,
        "--ft_mode",
        type=str,
        default="staged_reg_es",
        choices=[
            "full",
            "early_stopping",
            "regularized",
            "predictor_only",
            "private_only",
            "staged_reg_es",
        ],
        help="Target city fine-tuning strategy.",
    )

    # ============================================================
    # Regularized fine-tuning arguments
    # ============================================================

    upsert_argument(
        parser,
        "--reg_lambda",
        type=float,
        default=1.0,
        help=(
            "Regularization weight. With reg_normalize=1, try 0.1 / 1.0 / 10.0."
        ),
    )

    upsert_argument(
        parser,
        "--reg_exclude_predictor",
        type=int,
        default=1,
        help=(
            "1: do not regularize stPredictor. "
            "0: regularize all trainable parameters."
        ),
    )

    upsert_argument(
        parser,
        "--reg_normalize",
        type=int,
        default=1,
        help=(
            "1: use average squared parameter distance. "
            "0: use raw sum of squared parameter distance."
        ),
    )

    # ============================================================
    # Layer-wise learning rate arguments
    # ============================================================

    upsert_argument(
        parser,
        "--predictor_lr_mult",
        type=float,
        default=1.0,
        help="Learning rate multiplier for stPredictor.",
    )

    upsert_argument(
        parser,
        "--private_lr_mult",
        type=float,
        default=0.5,
        help="Learning rate multiplier for spatialModel and stPare.",
    )

    upsert_argument(
        parser,
        "--other_lr_mult",
        type=float,
        default=0.25,
        help="Learning rate multiplier for other trainable parameters.",
    )

    # ============================================================
    # Staged strategy arguments
    # ============================================================

    upsert_argument(
        parser,
        "--warmup_epochs",
        type=int,
        default=5,
        help="Predictor-only warmup epochs for ft_mode=staged_reg_es.",
    )

    # ============================================================
    # Early stopping arguments
    # ============================================================

    upsert_argument(
        parser,
        "--es_patience",
        type=int,
        default=10,
        help="Alias for early stopping patience.",
    )

    upsert_argument(
        parser,
        "--es_val_ratio",
        type=float,
        default=0.2,
        help="Alias for target validation ratio.",
    )

    upsert_argument(
        parser,
        "--early_stop_patience",
        type=int,
        default=None,
        help="Stop if validation loss does not improve for this many epochs.",
    )

    upsert_argument(
        parser,
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum validation loss improvement to reset early stopping counter.",
    )

    upsert_argument(
        parser,
        "--early_stop_print_every",
        type=int,
        default=1,
        help="Print target fine-tuning loss every N epochs.",
    )

    # ============================================================
    # Target scarce data split arguments
    # ============================================================

    upsert_argument(
        parser,
        "--target_days",
        type=int,
        default=3,
        help="How many early days are used as scarce target-city data.",
    )

    upsert_argument(
        parser,
        "--target_train_ratio",
        type=float,
        default=None,
        help="Train ratio for scarce target-city data.",
    )

    upsert_argument(
        parser,
        "--target_val_ratio",
        type=float,
        default=None,
        help="Validation ratio for scarce target-city data.",
    )

    upsert_argument(
        parser,
        "--target_test_ratio",
        type=float,
        default=0.2,
        help="Test ratio for scarce target-city data.",
    )

    parsed = parser.parse_args()
    options = parsed.__dict__

    # ============================================================
    # Parameter alias mapping
    # ============================================================

    if options.get("early_stop_patience") is None:
        options["early_stop_patience"] = options.get("es_patience", 10)

    if options.get("target_val_ratio") is None:
        options["target_val_ratio"] = options.get("es_val_ratio", 0.2)

    if options.get("target_test_ratio") is None:
        options["target_test_ratio"] = 0.2

    if options.get("target_train_ratio") is None:
        options["target_train_ratio"] = (
            1.0
            - float(options["target_val_ratio"])
            - float(options["target_test_ratio"])
        )

    ratio_sum = (
        float(options["target_train_ratio"])
        + float(options["target_val_ratio"])
        + float(options["target_test_ratio"])
    )

    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            "target_train_ratio + target_val_ratio + target_test_ratio must be 1.0. "
            f"Current sum = {ratio_sum}"
        )

    # ============================================================
    # Seed
    # ============================================================

    os.environ["PYTHONHASHSEED"] = str(options["seed"])

    np.random.seed(1 + options["seed"])
    torch.manual_seed(12 + options["seed"])
    random.seed(1234 + options["seed"])

    if str(options["device"]).startswith("cuda"):
        torch.cuda.manual_seed_all(123 + options["seed"])
        torch.backends.cudnn.deterministic = True

    trainer_class = pFedCTP_ES

    return options, trainer_class


def main():
    options, trainer_class = read_options()
    trainer = trainer_class(options)
    trainer.train()


if __name__ == "__main__":
    main()