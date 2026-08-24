# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import tempfile

from vllm.logger import init_logger

logger = init_logger(__name__)


def get_distributed_init_method() -> str:
    """Return a fresh ``file://`` init_method for a diffusion worker's own process group.

    This is always intra-node: diffusion workers rendezvous over ``localhost`` only.
    Coordinating through a unique filesystem path instead of a pre-agreed TCP port
    avoids the bind-time EADDRINUSE race that port allocation is prone to.
    """
    with tempfile.NamedTemporaryFile(prefix="vllm_omni_dist_init_") as f:
        return f"file://{f.name}"
