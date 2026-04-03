# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import pytest

from shopOps.models import ShopopsAction
from shopOps.server.shopOps_environment import ShopopsEnvironment

logger = logging.getLogger(__name__)


def test_observation_schema_fields():
    env = ShopopsEnvironment(debug_mode=False)
    obs = env.reset(seed=123, tier="easy", split="train")

    logger.info("observation=%s", obs.model_dump())

    assert obs.case is not None
    assert obs.resources is not None
    assert obs.case_index == 0
    assert obs.step_index == 0
    assert obs.episode_id
    assert obs.tier == "easy"


def test_invalid_action_enum_rejected():
    with pytest.raises(ValueError):
        ShopopsAction(action_type="fly_to_moon")  # type: ignore[arg-type]
