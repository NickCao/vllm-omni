# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.cpu_model_runner import CPUModelRunner

from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner


class OmniCPUModelRunner(CPUModelRunner, OmniGPUModelRunner):
    def load_model(self, *args, **kwargs) -> None:
        CPUModelRunner.load_model(self, *args, **kwargs)
        self._omni_post_load_model()
