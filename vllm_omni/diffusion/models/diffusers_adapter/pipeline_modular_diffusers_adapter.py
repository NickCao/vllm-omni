# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Modular Diffusers adapter for vLLM-Omni.

Black-box wrapper around diffusers' ``ModularPipeline``, mirroring
``DiffusersAdapterPipeline`` but loading through the modular-pipeline
framework instead of the classic ``DiffusionPipeline``. Delegates full
pipeline execution to ``ModularPipeline.__call__()`` via
``DiffusersCallDelegationMixin``.

Also supports per-request step-wise execution (``SupportsStepExecution``):
every ``LoopSequentialPipelineBlocks`` instance anywhere in the pipeline's
full block tree is patched once, at load time, so its denoise loop pauses
after every iteration -- see ``_patch_loop_step_to_yield`` for the mechanism.
Each request then simply calls the pipeline normally, inside a dedicated
``greenlet``; diffusers' own ``ConditionalPipelineBlocks`` dispatch picks the
right (already-patched) workflow variant live, exactly as it would for a
monolithic call. This gives per-timestep control within a single request
(responsive cancellation, streaming partial output) without reimplementing
any model's denoise loop.

Concurrent, interleaved requests are refused (``max_num_seqs`` must be 1).
At least two shared diffusers components -- ``scheduler`` (internal step
index) and ``guider`` (``set_state(step=i, ...)``, confirmed universal
across every modular pipeline model) -- hold per-step mutable state as plain
instance attributes on the pipeline; interleaving requests would mean
isolating every one of those per request, and there's no guarantee some
other model-specific block doesn't hold similar state we haven't found.
Refusing concurrency sidesteps that whole problem rather than chasing it
case by case. This also means step execution here does NOT provide
cross-request GPU-batched compute or head-of-line-blocking avoidance -- each
request still drives its own independent, unbatched diffusers call, one at
a time.

Does not support CFG parallel, sequence parallel, or TeaCache / Cache-DiT.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

import torch
from diffusers.modular_pipelines import ComponentsManager, ModularPipeline
from diffusers.modular_pipelines.modular_pipeline import LoopSequentialPipelineBlocks, ModularPipelineBlocks
from greenlet import greenlet
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.diffusers_adapter.call_delegation_mixin import DiffusersCallDelegationMixin
from vllm_omni.diffusion.models.diffusers_adapter.pipeline_utils import BasePipelineUtils, get_pipeline_utils
from vllm_omni.diffusion.models.diffusers_adapter.quantization_utils import (
    apply_diffusers_quantization_config,
    convert_diffusers_quantization_config,
)
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import StepRequestState

logger = logging.getLogger(__name__)


class ModularDiffusersAdapterPipeline(nn.Module, DiffusionPipelineProfilerMixin, DiffusersCallDelegationMixin):
    """Black-box adapter that delegates full pipeline execution to a ``ModularPipeline``.

    Usage::

        adapter = ModularDiffusersAdapterPipeline(od_config=od_config)
        adapter.load_weights()  # calls ModularPipeline.from_pretrained()
        output = adapter.forward(req)
    """

    supports_request_batch = False
    supports_step_execution: bool = True

    def __init__(self, *, od_config: OmniDiffusionConfig, device: torch.device | None = None):
        super().__init__()
        # Interleaving requests would mean isolating every shared, per-step-
        # mutable diffusers component per request -- `scheduler` (internal step
        # index) and `guider` (`set_state(step=i, ...)`) are confirmed, but
        # there's no guarantee some other model-specific block doesn't hold
        # similar state we haven't found. Refuse interleaving for now rather
        # than risk silent cross-request corruption under concurrent load --
        # `max_num_seqs` already defaults to 1, this just makes going above
        # that an explicit error instead of a silent footgun.
        if od_config.step_execution and od_config.max_num_seqs > 1:
            raise NotImplementedError(
                "ModularDiffusersAdapterPipeline's step execution does not support "
                f"max_num_seqs > 1 (got {od_config.max_num_seqs}): interleaving "
                "concurrent requests risks corrupting per-request state in "
                "diffusers components we haven't audited for every model."
            )
        self._pipeline: ModularPipeline
        self._manager: ComponentsManager
        self._accept_call_kwargs: set[str] | None = None  # ModularPipeline.__call__ accepts any kwargs
        self.od_config = od_config
        self.device = device
        self._pipeline_utils: BasePipelineUtils = BasePipelineUtils()

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler,
            profiler_targets=["forward"],
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self) -> None:
        """Load the diffusers pipeline via ``ModularPipeline.from_pretrained()``."""
        model_id = self.od_config.model
        dtype = self.od_config.dtype

        load_kwargs = {
            "torch_dtype": dtype,
            **self.od_config.diffusers_load_kwargs,
        }
        convert_diffusers_quantization_config(load_kwargs)

        pipeline_class = self.od_config.diffusers_pipeline_cls
        pipeline_class_name = pipeline_class.__name__ if pipeline_class is not None else None
        self._pipeline_utils = get_pipeline_utils(pipeline_class_name)
        self._pipeline_utils.update_load_kwargs(self.od_config, load_kwargs)

        self._manager = ComponentsManager()
        self._pipeline = ModularPipeline.from_pretrained(
            model_id,
            components_manager=self._manager,
            **load_kwargs,
        )

        if self.od_config.quantization_config is not None and "quantization_config" not in load_kwargs:
            apply_diffusers_quantization_config(self.od_config, load_kwargs, self._pipeline.components)

        self._pipeline.load_components(**load_kwargs)
        self._pipeline.to(self.device)
        self._pipeline_utils.apply_post_load_updates(self._pipeline, self.od_config)

        # VAE slicing and tiling
        vae = getattr(self._pipeline, "vae", None)
        if vae is not None:
            if self.od_config.vae_use_slicing and hasattr(vae, "enable_slicing"):
                vae.enable_slicing()
            if self.od_config.vae_use_tiling and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()

        # Attention backend
        self._set_attention_backend()

        # Step-wise execution: patch every denoise loop in the pipeline's full
        # block tree once, up front. See `_collect_loop_blocks` for why there
        # is one such instance per workflow variant, not just one overall.
        loop_blocks = self._collect_loop_blocks(self._pipeline._blocks)
        for loop_block in loop_blocks:
            self._patch_loop_step_to_yield(loop_block)

        logger.info(
            "Loaded modular pipeline %s (patched %d denoise loop(s) for step execution).",
            self._pipeline.__class__.__name__,
            len(loop_blocks),
        )

    # ------------------------------------------------------------------
    # Step-wise execution
    # ------------------------------------------------------------------
    #
    # Every denoise loop in the pipeline's full block tree is patched once,
    # at load time (see `load_weights`), so its `loop_step` pauses a dedicated
    # greenlet after every real iteration. Each request then just calls the
    # pipeline normally -- diffusers' own `ConditionalPipelineBlocks.__call__`
    # dispatch (a live, per-call lookup, unrelated to `get_execution_blocks`)
    # already picks the right, already-patched workflow variant on its own.
    # See `_patch_loop_step_to_yield` for why patching `loop_step` works
    # without reimplementing any model's denoise loop, and why a plain
    # exception-based pause/restart does not (it corrupts setup-time math
    # that assumes the full, untruncated timestep schedule).

    @classmethod
    def _collect_loop_blocks(cls, blocks: ModularPipelineBlocks) -> list[LoopSequentialPipelineBlocks]:
        """Recursively find every `LoopSequentialPipelineBlocks` in a block tree.

        The full, unresolved tree has one loop-block *instance* per workflow
        variant (e.g. QwenImage's "denoise" `ConditionalPipelineBlocks` holds a
        distinct object for each of text2image/inpaint/controlnet/etc.), unlike
        a resolved per-request graph, which has exactly one. All of them must
        be patched up front, since which one actually runs for a given request
        is decided live by `ConditionalPipelineBlocks.__call__`, not by us.
        """
        if isinstance(blocks, LoopSequentialPipelineBlocks):
            return [blocks]
        found: list[LoopSequentialPipelineBlocks] = []
        for sub_block in (getattr(blocks, "sub_blocks", None) or {}).values():
            found.extend(cls._collect_loop_blocks(sub_block))
        return found

    @staticmethod
    def _patch_loop_step_to_yield(loop_block: LoopSequentialPipelineBlocks) -> None:
        """Make every `loop_step()` call pause the current greenlet after doing real work.

        `loop_step` is defined once, generically, on the `LoopSequentialPipelineBlocks`
        base class (only `__call__`'s setup/teardown is model-specific), so this is a
        safe, model-independent interception point. Instance-level assignment (mirroring
        `HookRegistry`'s identical forward-patching pattern for offload hooks) bypasses
        the descriptor/auto-binding protocol, so the wrapper must accept the same
        (components, state, **kwargs) signature `loop_step` normally has.

        `greenlet.getcurrent().parent` is always the greenlet that created the current
        one -- i.e. whichever request-processing call (`prepare_encode`/`denoise_step`/
        `post_decode`) is driving this request, regardless of which request it is.
        """
        original_loop_step = loop_block.loop_step  # bound method, inherited base-class impl

        @functools.wraps(original_loop_step)
        def wrapped(components, state, **kwargs):
            components, state = original_loop_step(components, state, **kwargs)
            greenlet.getcurrent().parent.switch(kwargs.get("i"), kwargs.get("t"))
            return components, state

        loop_block.loop_step = wrapped

    def _build_step_call_kwargs(self, state: StepRequestState) -> dict[str, Any]:
        """Build diffusers `__call__` kwargs for one request.

        A single-request counterpart to `DiffusersCallDelegationMixin._build_call_kwargs`
        (which is batch-shaped, reading from a `DiffusionRequestBatch`); this reads
        from `StepRequestState`/its `sampling` params instead. `_accept_call_kwargs`
        is always None for this adapter, so unlike the mixin there is no accept-list
        filtering to replicate here.
        """
        sampling = state.sampling
        input_kwargs = self._extract_input([state.prompt] if state.prompt is not None else [""])

        kwargs: dict[str, Any] = dict(self.od_config.diffusers_call_kwargs)
        kwargs.update(input_kwargs)

        for key, value in sampling.__dict__.items():
            if value is not None:
                kwargs[key] = value

        if output_type := sampling.output_type or self.od_config.output_type:
            kwargs["output_type"] = output_type

        if sampling.generator is not None:
            kwargs["generator"] = sampling.generator
        elif sampling.seed is not None:
            kwargs["generator"] = torch.Generator(device=sampling.generator_device).manual_seed(sampling.seed)
        else:
            kwargs["generator"] = torch.Generator(device=sampling.generator_device)

        return kwargs

    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState:
        """Wire up this request's worker greenlet.

        No diffusers work happens here -- the worker greenlet is created but not
        started. The first `denoise_step()` call runs it up to its first pause
        point, which covers text encoding and latent prep too: there is no
        separate "encode-only" pause point in the pipeline's block graph. Which
        workflow variant actually runs (text2image/inpaint/controlnet/etc.) is
        decided live, per request, by diffusers' own dispatch -- see
        `load_weights`/`_collect_loop_blocks` for why every variant's loop is
        already patched by this point.
        """
        del kwargs
        call_kwargs = self._build_step_call_kwargs(state)
        pipe = self._pipeline

        # No per-request isolation needed for stateful shared components
        # (scheduler's internal step index, guider's set_state(step=i, ...)):
        # __init__ refuses max_num_seqs > 1 for this adapter, so exactly one
        # request is ever in flight and there's nothing else to collide with.
        def run_pipeline():
            with torch.inference_mode():
                return pipe(**call_kwargs)

        num_inference_steps = int(call_kwargs.get("num_inference_steps") or state.sampling.num_inference_steps or 1)
        num_images_per_prompt = int(call_kwargs.get("num_images_per_prompt") or 1)

        # These placeholders only need to satisfy the runner's own batching/
        # completion bookkeeping (StepRequestState.denoise_completed,
        # InputBatch.make_batch's non-None / shape checks). The actual latents
        # never leave the paused greenlet's local state, so the runner has no
        # use for real values here -- only the shapes/lengths matter.
        state.timesteps = torch.arange(num_inference_steps)
        state.step_index = 0
        state.latents = torch.zeros(num_images_per_prompt, 1, device=self.device)

        state.extra["worker_greenlet"] = greenlet(run_pipeline)
        state.extra["final_output"] = None
        state.extra["error"] = None

        return state

    def _resume(self, state: StepRequestState) -> Any:
        """Resume this request's worker greenlet by exactly one pause point.

        Exceptions raised inside the worker propagate out of `switch()` into this
        call; they're caught here and stashed on `state` rather than left to
        crash the whole engine tick.
        """
        worker: greenlet = state.extra["worker_greenlet"]
        try:
            return worker.switch()
        except Exception as exc:  # noqa: BLE001 -- surface as a request-level error, not an engine crash
            logger.exception("Modular diffusers step execution failed for request %s", state.request_id)
            state.extra["error"] = exc
            return None

    def denoise_step(
        self,
        input_batch: InputBatch,
        *,
        states: Sequence[StepRequestState] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | None:
        """Advance every active request's worker greenlet by exactly one iteration.

        Always returns None: unlike native pipelines there is no combined batched
        tensor to hand back (each request drives its own independent, unbatched
        diffusers call) -- all real work happens here, `step_scheduler()` only
        advances bookkeeping.
        """
        del input_batch, kwargs
        for state in states or ():
            worker: greenlet = state.extra.get("worker_greenlet")
            if worker is None or worker.dead or state.extra.get("error") is not None:
                continue
            result = self._resume(state)
            if worker.dead:
                state.extra["final_output"] = result
        return None

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor | None, **kwargs: Any) -> None:
        """Advance step bookkeeping. The scheduler update already happened inside
        `denoise_step()`'s `loop_step()` call, so there is nothing left to do here
        except advance `step_index` (or fast-track to `post_decode()` on error).
        """
        del noise_pred, kwargs
        if state.extra.get("error") is not None:
            state.step_index = state.total_steps
            return
        state.step_index += 1

    def post_decode(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:
        """Drive the worker greenlet through decode (if not already finished) and
        wrap its result.

        The greenlet's last `loop_step()` pause leaves it just before the
        pipeline's after-loop/decode/postprocess blocks; resuming once more runs
        the rest of the pipeline unmodified -- diffusers does the actual VAE
        decode here, not us.
        """
        del kwargs
        error = state.extra.get("error")
        if error is not None:
            return DiffusionOutput(error=str(error))

        final_output = state.extra.get("final_output")
        if final_output is None:
            final_output = self._resume(state)
            if state.extra.get("error") is not None:
                return DiffusionOutput(error=str(state.extra["error"]))
            worker: greenlet = state.extra["worker_greenlet"]
            if not worker.dead:
                return DiffusionOutput(error=f"Unexpected extra pause point for request {state.request_id}")

        return self._wrap_output(final_output)
