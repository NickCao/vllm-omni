# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for OmniDiffusionConfig distributed_init_method resolution.

Diffusion workers rendezvous over torch.distributed intra-node only, so this
is resolved via a fresh file:// path rather than a TCP port (see issue #3794
and the follow-up port-race elimination in #6425): no port to pick means no
bind-time EADDRINUSE race to retry around.
"""

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestOmniDiffusionConfigDistributedInitMethod:
    def test_resolves_to_a_file_init_method(self) -> None:
        config = OmniDiffusionConfig(model="test")
        assert config.distributed_init_method is not None
        assert config.distributed_init_method.startswith("file://")

    def test_each_config_gets_a_unique_init_method(self) -> None:
        methods = {OmniDiffusionConfig(model="test").distributed_init_method for _ in range(5)}
        assert len(methods) == 5
