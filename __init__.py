# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shopops Environment."""

# Allow importing this module both as a package (relative imports)
# and as a top-level module (pytest sometimes imports repo-root __init__.py).
if __package__:
    from .client import ShopopsEnv
    from .models import (
        ActionType,
        CaseType,
        CustomerTier,
        DeliveryStatus,
        EscalationReason,
        FraudSignal,
        IssueSeverity,
        ItemCategory,
        ShopopsAction,
        ShopopsObservation,
    )
else:
    from client import ShopopsEnv
    from models import (
        ActionType,
        CaseType,
        CustomerTier,
        DeliveryStatus,
        EscalationReason,
        FraudSignal,
        IssueSeverity,
        ItemCategory,
        ShopopsAction,
        ShopopsObservation,
    )

__all__ = [
    "ShopopsAction",
    "ShopopsObservation",
    "ShopopsEnv",
    "ActionType",
    "CaseType",
    "CustomerTier",
    "DeliveryStatus",
    "EscalationReason",
    "FraudSignal",
    "IssueSeverity",
    "ItemCategory",
]
