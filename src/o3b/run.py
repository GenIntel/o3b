from __future__ import annotations


def _requests_cuda(cfg) -> bool:
    """True if any ``device: cuda…`` entry appears anywhere in the config."""
    if isinstance(cfg, dict):
        for key, value in cfg.items():
            if (isinstance(key, str) and "device" in key
                    and isinstance(value, str) and value.startswith("cuda")):
                return True
            if _requests_cuda(value):
                return True
        return False
    if isinstance(cfg, (list, tuple)):
        return any(_requests_cuda(v) for v in cfg)
    return False


def _run_bench_run_with_cfg(run_raw: dict, run_name: str) -> None:
    """Execute one benchmark evaluation pass given a fully-resolved config dict.

    Args:
        run_raw:  Merged config dict. ``run_raw["dataset"]`` must already be the
                  fully-merged dataset config (base defaults + benchmark + ablation).
        run_name: Used as the W&B run name and as the fallback W&B project name.
    """
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf

    from o3b.dataset.dataset import DatasetConfig, build_dataset, ItemType
    from o3b.task.task import build_task
    from o3b.data.datatypes.object import collate_object_pairs
    from o3b.data.datatypes.frame_object import collate_frame_object_pairs
    from o3b import ddp

    # ── distributed (torchrun) ────────────────────────────────────────────────
    # No-op unless WORLD_SIZE > 1, i.e. unless the job preamble launched this
    # through torchrun (platform.gpu_count_per_node / node_count > 1).  Both
    # halves of a run are data-parallel the same way (o3b.ddp): every rank
    # builds the method and runs train_method, then every rank evaluates its
    # own slice of the eval set and the metrics are reduced once at the end.
    # Eval needs no collective while it runs — there are no gradients — so the
    # ranks only meet again in that final reduction.
    rank, world_size, _ = ddp.init_distributed()
    is_main = ddp.is_main_process(rank)
    distributed = ddp.is_initialized()
    if world_size > 1:
        print(f"DDP:     rank {rank}/{world_size} (local_rank {ddp.get_local_rank()})")

    # ── fail fast on a GPU-less environment ───────────────────────────────────
    # Nothing recovers from a missing GPU: the method would crash in .to(device),
    # get swallowed by the fallback below and the whole benchmark would run as
    # GT/oracle — burning the allocation and reporting misleading numbers.
    if _requests_cuda(run_raw):
        import os
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "config requests a CUDA device but no CUDA GPU is available "
                f"(CUDA_VISIBLE_DEVICES="
                f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}) — "
                "ending the run instead of continuing with partial init."
            )

    eval_cfg    = run_raw.get("eval") or {}
    batch_size  = eval_cfg.get("batch_size", 4)
    num_workers = int(eval_cfg.get("num_workers", 4))

    # Every rank builds the eval dataset, its loader and the task — each one
    # evaluates a slice of the dataset (`sampler` below).  rank 0 goes first:
    # a cold sharded cache is *built* by the first process that opens it, and
    # several ranks writing the same Arrow shards would corrupt them. The
    # others wait here and then find it warm.
    if distributed and not is_main:
        ddp.dist_sync_processes()
    dataset_cfg = DatasetConfig.from_dict(run_raw["dataset"])
    dataset     = build_dataset(dataset_cfg)
    if distributed and is_main:
        ddp.dist_sync_processes()
    if is_main:
        print(f"Dataset: {dataset_cfg.class_name}  ({len(dataset)} items)")

    collate_fn = (collate_frame_object_pairs
                  if dataset_cfg.item_type == ItemType.FRAME_OBJECT_PAIR
                  else collate_object_pairs)
    # rank r evaluates items r, r + W, r + 2W, …  Unlike DistributedSampler
    # this pads nothing, so the union over the ranks is exactly the dataset and
    # no item is scored twice — the shard sizes then differ by at most one,
    # which is fine because nothing inside the loop is collective.
    sampler = (list(range(rank, len(dataset), world_size)) if distributed
               else None)
    loader = DataLoader(
        dataset,
        # per-rank, like the training batch size: the number of items in flight
        # grows with the number of GPUs
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    task_cfg = OmegaConf.create(run_raw["task"])
    task     = build_task(task_cfg)
    if is_main:
        print(f"Task:    {run_raw['task']['class_name']}")

    # ── method (optional) ─────────────────────────────────────────────────────
    # The method runs on each batch before the task (e.g. a pose estimator that
    # writes predicted poses). A method that fails to build ends the run: the
    # alternative is to evaluate the task on the raw batch (GT/oracle), which
    # produces a full set of plausible metrics recorded under the method's name.
    # That silently misreported four JUPITER runs as `morpheus` (missing open3d,
    # a cold torch.hub cache, a stale nvdiffrast) before anyone noticed.
    # Evaluating the oracle is still reachable, but only on purpose:
    #   allow_oracle_fallback: true   in the run config, or
    #   O3B_ALLOW_ORACLE_FALLBACK=1   in the environment.
    method = None
    method_cfg = run_raw.get("method")
    if method_cfg:
        cls_name = method_cfg.get("class_name")
        try:
            from housecorr3dv2.method.method import build_method, MethodConfig
            method = build_method(MethodConfig.from_dict(dict(method_cfg)))
            if is_main:
                print(f"Method:  {cls_name}")
        except Exception as exc:
            import os, traceback
            traceback.print_exc()
            allow = bool(run_raw.get("allow_oracle_fallback", False)) or \
                os.environ.get("O3B_ALLOW_ORACLE_FALLBACK", "").lower() in ("1", "true", "yes")
            if not allow:
                raise RuntimeError(
                    f"could not build method {cls_name!r}: {exc}\n"
                    f"Ending the run. Falling back to the task on the raw batch "
                    f"would report GT/oracle numbers under the name {cls_name!r}. "
                    f"To evaluate the oracle deliberately, set "
                    f"allow_oracle_fallback: true in the run config or "
                    f"O3B_ALLOW_ORACLE_FALLBACK=1 in the environment."
                ) from exc
            print(f"WARNING: could not build method {cls_name!r} ({exc}); "
                  f"running task on raw batch (GT/oracle) because the fallback "
                  f"was explicitly enabled. Results are NOT {cls_name!r}.")

    # ── wandb init (before training so per-batch train losses are logged) ─────
    # Rank 0 owns the run: every rank logging into the same project would
    # produce world_size runs whose train/* curves interleave at random.  The
    # per-batch train/* and batch/* curves are therefore rank 0's alone; the
    # eval/* summary at the end is reduced over every rank.
    _wb = None
    wandb_cfg = run_raw.get("wandb") or {}
    if wandb_cfg is not False and is_main:
        try:
            import wandb as _wb_mod
            wb_project = wandb_cfg.get("project", run_name)
            _wb_mod.init(
                project=wb_project,
                name=run_name,
                config=run_raw,
                reinit=True,
            )
            _wb = _wb_mod
            print(f"W&B:     project={wb_project}  run={run_name}")
        except ImportError:
            print("INFO: wandb not installed — skipping W&B logging")

    # ── training (optional) ───────────────────────────────────────────────────
    # Methods exposing train_method train themselves before evaluation:
    # needs_training() decides (train.dataset_train configured and the loaded
    # checkpoint's epoch below train.epochs), and the method creates its
    # training dataset lazily from that config, like its pose model. Batch
    # losses are logged to the active wandb run under train/.
    if method is not None and hasattr(method, "train_method"):
        needs_training = getattr(method, "needs_training", None)
        if callable(needs_training) and needs_training():
            if is_main:
                print("Train:   running method training")
            method.train_method()
        elif is_main:
            print("Train:   skipped (no train dataset or checkpoint fully trained)")

    # Training left the ranks at different points in time (rank 0 also wrote a
    # checkpoint); meet once before they go their own way through eval.
    ddp.dist_sync_processes()

    if is_main:
        print(f"Eval:    batch_size={batch_size}  n_batches={len(loader)}"
              + (f"  (per rank, {world_size} ranks)" if distributed else "")
              + "\n")

    accum: dict[str, list] = {}
    n_samples = 0
    qualit_log_batches = eval_cfg.get("qualit_log_batches", 8)

    # ── cost metrics ──────────────────────────────────────────────────────────
    # time_method_s_per_sample times only the method's forward pass (the number
    # that belongs in a runtime comparison), time_total_s the whole pass incl.
    # dataloading.  The peak allocator stats are reset here, after training, so
    # eval memory is not attributed the (much larger) training peak.
    #
    # All CUDA calls are scoped to the method's own device (its `device:`
    # config, cuda:0 for methods that do not expose one): the no-arg forms act
    # on the *current* device and would silently measure cuda:0 while the
    # method ran on cuda:1.  Iterating over all visible devices instead is not
    # an option — synchronizing a device creates a CUDA context on it.
    #
    # gpu_mem_peak_gb is the allocator's peak *allocated*, _reserved_gb its
    # peak *reserved* from the driver (larger by caching/fragmentation, and the
    # number that decides which GPU the run fits on).  Both miss memory taken
    # outside the torch allocator: the CUDA context and nvdiffrast's buffers.
    import time
    import torch as _torch

    _mem_dev = None
    if _torch.cuda.is_available():
        _dev_cfg = getattr(method, "device", None) if method is not None else None
        try:
            _mem_dev = _torch.device(_dev_cfg) if _dev_cfg is not None else None
        except (TypeError, ValueError, RuntimeError):
            _mem_dev = None
        if _mem_dev is None or _mem_dev.type != "cuda":
            _mem_dev = _torch.device("cuda", _torch.cuda.current_device())

    def _cuda_mem(fn):
        """fn(device) → bytes, or None if that device has no CUDA context yet."""
        if _mem_dev is None:
            return None
        try:
            return fn(_mem_dev)
        except RuntimeError:
            return None  # device never initialised → nothing was allocated

    _cuda_mem(_torch.cuda.reset_peak_memory_stats)
    t_method_total = 0.0
    t_eval_start = time.perf_counter()

    from tqdm import tqdm
    bar = tqdm(loader, total=len(loader), unit="batch", desc="eval",
               disable=not is_main)
    for batch_idx, batch in enumerate(bar):
        return_qualit = (_wb is not None) and (batch_idx < qualit_log_batches)

        method_qualit = None
        if method is not None:
            # synchronize on both sides: CUDA kernels are launched
            # asynchronously, so without this we would time launches, not work
            if _mem_dev is not None:
                _torch.cuda.synchronize(_mem_dev)
            _t_method_start = time.perf_counter()
            result = method(batch, return_qualit=return_qualit)
            if _mem_dev is not None:
                _torch.cuda.synchronize(_mem_dev)
            t_method_total += time.perf_counter() - _t_method_start
            if isinstance(result, tuple):
                batch, method_qualit = result
            else:
                batch = result

        quant, qualit = task(batch, return_qualit=return_qualit)

        B = (batch.src_obj_kpts3d.shape[0]
             if batch.src_obj_kpts3d is not None else batch_size)
        n_samples += B

        for metric_name, value in quant.mean().items():
            accum.setdefault(metric_name, []).append(value)

        if _wb is not None:
            wb_log = quant.to_wandb_log(prefix="batch", wb=_wb)
            if qualit is not None:
                wb_log.update(qualit.to_wandb_log(prefix="qualit", wb=_wb, log_imgs=True))
            if method_qualit is not None:
                for k, v in method_qualit.items():
                    import numpy as np
                    wb_log[k] = _wb.Image(v) if isinstance(v, np.ndarray) else v
            wb_log["batch/n_samples"] = n_samples
            # no explicit step: training already advanced the global wandb
            # step, and steps must be monotonically increasing
            _wb.log(wb_log)

        bar.set_postfix({"samples": n_samples,
                         **{k: round(sum(v) / len(v), 4) for k, v in accum.items()}})

    t_eval_total   = time.perf_counter() - t_eval_start
    _peak_alloc    = _cuda_mem(_torch.cuda.max_memory_allocated)
    _peak_resv     = _cuda_mem(_torch.cuda.max_memory_reserved)

    # ── reduce over the ranks ─────────────────────────────────────────────────
    # This is the only collective of the eval pass, so an imbalanced shard only
    # costs the others a wait here, never a timeout mid-loop.
    #
    # A metric is reported as the mean of its per-batch means, so summing the
    # values and their counts per key and dividing afterwards gives exactly
    # what one process over the same batches would report. The keys are unioned
    # rather than assumed equal: `quant.mean()` drops a metric whose batch held
    # no finite value, and a whole rank can miss one that way.
    # (Batches are cut differently for a different rank count, so a metric
    # still moves in the last decimals between 1 and N GPUs — the same effect a
    # changed eval.batch_size already has.)
    if distributed:
        local_sums = {k: (float(sum(v)), len(v)) for k, v in accum.items()}
        accum_reduced: dict[str, tuple[float, int]] = {}
        for rank_sums in ddp.all_gather_object(local_sums):
            for k, (s, n) in rank_sums.items():
                s0, n0 = accum_reduced.get(k, (0.0, 0))
                accum_reduced[k] = (s0 + s, n0 + n)
        metrics = {k: s / n for k, (s, n) in accum_reduced.items() if n}
        n_samples      = int(ddp.all_reduce_sum(n_samples))
        t_method_total = ddp.all_reduce_sum(t_method_total)
        # the pass lasts as long as its slowest rank, and has to fit in the
        # memory of its hungriest one — a mean would hide both
        t_eval_total   = ddp.all_reduce_max(t_eval_total)
        # whether there is a memory number at all is decided jointly: a rank
        # that skipped the reduction because its device never got a CUDA
        # context would leave the others waiting in it
        if ddp.all_reduce_max(0.0 if _peak_alloc is None else 1.0) > 0:
            _peak_alloc = ddp.all_reduce_max(float(_peak_alloc or 0.0))
            _peak_resv  = ddp.all_reduce_max(float(_peak_resv or 0.0))
        else:
            _peak_alloc = _peak_resv = None
    else:
        metrics = {k: sum(v) / len(v) for k, v in accum.items()}

    # nothing collective happens after this point
    ddp.cleanup_process_group()
    if not is_main:
        return

    cost_metrics = {"time_total_s": t_eval_total}
    if method is not None:
        cost_metrics["time_method_s_per_sample"] = t_method_total / max(n_samples, 1)
    if _peak_alloc is not None:
        cost_metrics["gpu_mem_peak_gb"] = _peak_alloc / 2 ** 30
        cost_metrics["gpu_mem_peak_reserved_gb"] = _peak_resv / 2 ** 30
    if distributed:
        cost_metrics["world_size"] = float(world_size)
        cost_metrics["batch_size_global"] = float(batch_size * world_size)

    print(f"\n{'─'*50}")
    print(f"Results  ({n_samples} samples)")
    for k, v in metrics.items():
        print(f"  {k:<25} {v:.4f}")
    for k, v in cost_metrics.items():
        print(f"  {k:<25} {v:.4f}")

    if _wb is not None:
        final_metrics = {f"eval/{k}": v for k, v in metrics.items()}
        final_metrics["eval/n_samples"] = n_samples
        final_metrics.update({f"eval/{k}": v for k, v in cost_metrics.items()})
        _wb.log(final_metrics)
        _wb.finish()
