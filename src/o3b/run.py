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

    dataset_cfg = DatasetConfig.from_dict(run_raw["dataset"])
    dataset     = build_dataset(dataset_cfg)
    print(f"Dataset: {dataset_cfg.class_name}  ({len(dataset)} items)")

    eval_cfg    = run_raw.get("eval") or {}
    batch_size  = eval_cfg.get("batch_size", 4)
    num_workers = int(eval_cfg.get("num_workers", 4))

    collate_fn = (collate_frame_object_pairs
                  if dataset_cfg.item_type == ItemType.FRAME_OBJECT_PAIR
                  else collate_object_pairs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    task_cfg = OmegaConf.create(run_raw["task"])
    task     = build_task(task_cfg)
    print(f"Task:    {run_raw['task']['class_name']}")

    # ── method (optional) ─────────────────────────────────────────────────────
    # The method runs on each batch before the task (e.g. a pose estimator that
    # writes predicted poses). If it cannot be built (e.g. missing dependency)
    # we warn and fall back to the task on the raw batch (GT/oracle) — except
    # for CUDA failures (no GPU, driver error, OOM), which are environment
    # problems the fallback would only hide: those end the run.
    method = None
    method_cfg = run_raw.get("method")
    if method_cfg:
        cls_name = method_cfg.get("class_name")
        try:
            from housecorr3dv2.method.method import build_method, MethodConfig
            method = build_method(MethodConfig.from_dict(dict(method_cfg)))
            print(f"Method:  {cls_name}")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            if "CUDA" in str(exc):
                raise RuntimeError(
                    f"could not build method {cls_name!r}: CUDA failure "
                    f"({exc}) — ending the run instead of continuing with "
                    f"partial init."
                ) from exc
            print(f"WARNING: could not build method {cls_name!r} ({exc}); "
                  f"running task on raw batch (GT/oracle).")

    # ── wandb init (before training so per-batch train losses are logged) ─────
    _wb = None
    wandb_cfg = run_raw.get("wandb") or {}
    if wandb_cfg is not False:
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
            print("Train:   running method training")
            method.train_method()
        else:
            print("Train:   skipped (no train dataset or checkpoint fully trained)")

    print(f"Eval:    batch_size={batch_size}  n_batches={len(loader)}\n")

    accum: dict[str, list] = {}
    n_samples = 0
    qualit_log_batches = eval_cfg.get("qualit_log_batches", 8)

    # ── cost metrics ──────────────────────────────────────────────────────────
    # time_method_s_per_sample times only the method's forward pass (the number
    # that belongs in a runtime comparison), time_total_s the whole pass incl.
    # dataloading.  The peak allocator stats are reset here, after training, so
    # eval memory is not attributed the (much larger) training peak.
    import time
    import torch as _torch
    _cuda = _torch.cuda.is_available()
    if _cuda:
        _torch.cuda.reset_peak_memory_stats()
    t_method_total = 0.0
    t_eval_start = time.perf_counter()

    from tqdm import tqdm
    bar = tqdm(loader, total=len(loader), unit="batch", desc="eval")
    for batch_idx, batch in enumerate(bar):
        return_qualit = (_wb is not None) and (batch_idx < qualit_log_batches)

        method_qualit = None
        if method is not None:
            # synchronize on both sides: CUDA kernels are launched
            # asynchronously, so without this we would time launches, not work
            if _cuda:
                _torch.cuda.synchronize()
            _t_method_start = time.perf_counter()
            result = method(batch, return_qualit=return_qualit)
            if _cuda:
                _torch.cuda.synchronize()
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

    cost_metrics = {"time_total_s": time.perf_counter() - t_eval_start}
    if method is not None:
        cost_metrics["time_method_s_per_sample"] = t_method_total / max(n_samples, 1)
    if _cuda:
        # allocator peak only — excludes the CUDA context and any memory held
        # by non-PyTorch back-ends (nvdiffrast, pyrender), but it is the
        # reproducible number to compare methods with on a shared node
        cost_metrics["gpu_mem_peak_gb"] = _torch.cuda.max_memory_allocated() / 2 ** 30

    print(f"\n{'─'*50}")
    print(f"Results  ({n_samples} samples)")
    for k, vals in accum.items():
        print(f"  {k:<25} {sum(vals)/len(vals):.4f}")
    for k, v in cost_metrics.items():
        print(f"  {k:<25} {v:.4f}")

    if _wb is not None:
        final_metrics = {f"eval/{k}": sum(v) / len(v) for k, v in accum.items()}
        final_metrics["eval/n_samples"] = n_samples
        final_metrics.update({f"eval/{k}": v for k, v in cost_metrics.items()})
        _wb.log(final_metrics)
        _wb.finish()
