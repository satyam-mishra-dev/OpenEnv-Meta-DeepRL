# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shopops Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

if __package__:
    from .models import ShopopsAction, ShopopsObservation
else:
    from models import ShopopsAction, ShopopsObservation


class ShopopsEnv(
    EnvClient[ShopopsAction, ShopopsObservation, State]
):
    """
    Client for the Shopops Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with ShopopsEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(seed=42, tier="easy")
        ...     action = ShopopsAction(action_type=ActionType.REJECT)
        ...     result = client.step(action)
        ...     print(result.observation.case.case_id)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = ShopopsEnv.from_docker_image("shopOps-env:latest")
        >>> try:
        ...     result = client.reset(seed=1)
        ...     result = client.step(ShopopsAction(action_type=ActionType.REJECT))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: ShopopsAction) -> Dict:
        """
        Convert ShopopsAction to JSON payload for step message.

        Args:
            action: ShopopsAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[ShopopsObservation]:
        """
        Parse server response into StepResult[ShopopsObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with ShopopsObservation
        """
        obs_data = payload.get("observation", {})
        observation = ShopopsObservation.model_validate(obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
