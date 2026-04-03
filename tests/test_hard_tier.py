# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from shopOps.server.shopOps_environment import ShopopsEnvironment


def test_hard_tier_partial_observability():
    env = ShopopsEnvironment(debug_mode=True)
    obs = env.reset(seed=9001, tier="hard", split="test")

    case = obs.case
    # Exact values hidden in hard tier
    assert case.order_value_usd is None
    assert case.days_since_order is None

    # At least one of the normally visible hard-tier fields should be hidden
    hidden_candidates = [
        case.order_value_bucket,
        case.order_age_bucket,
        case.delivery_status,
        case.prior_refund_count_bucket,
    ]
    assert any(value is None for value in hidden_candidates)


def test_hard_tier_adversarial_seed():
    env = ShopopsEnvironment(debug_mode=True)
    obs = env.reset(seed=9001, tier="hard", split="test")

    assert env._adversarial_case_ids
    assert any(case_id.startswith("adv-") for case_id in env._adversarial_case_ids)
