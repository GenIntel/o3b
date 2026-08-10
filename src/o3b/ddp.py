"""Distributed data-parallel helpers.

Everything here is a no-op outside a torchrun launch: ``WORLD_SIZE`` is unset
(→ 1), so a single-GPU run takes exactly the same code path it always did.

Under torchrun the launcher exports RANK / LOCAL_RANK / WORLD_SIZE /
MASTER_ADDR / MASTER_PORT; the job preamble written by ``o3b.cli``
(`_srun_env_lines`) is what sets the master address/port up on slurm, so a
process only has to read the environment here.

The training loop in ``housecorr3dv2.method.morpheus`` averages gradients with
`average_gradients` instead of wrapping the model in ``DistributedDataParallel``:
its trainable parameters are spread over three unrelated objects (the feature
model, the DMTet objects3d, the instance-deform head), and which of them receive
gradients changes with the epoch (the NeMo two-stage schedule). DDP would need
one wrapper per object plus ``find_unused_parameters`` on each; an explicit
all-reduce over the optimizer's parameter list is equivalent (same average, one
collective per step) and indifferent to which loss ran.

Evaluation (``o3b.run``) is data-parallel the same way and for every method:
each rank runs the method over its own slice of the eval set — no gradients, so
nothing has to be synchronised while the loop runs — and the per-batch metrics
are reduced once at the end with `all_gather_object` / `all_reduce_sum` /
`all_reduce_max`.
"""

import os
from collections import defaultdict

import torch
import torch.distributed as dist


def get_world_size():
    return int(os.getenv("WORLD_SIZE", 1))


def get_rank():
    return int(os.getenv("RANK", 0))


def get_local_rank():
    return int(os.getenv("LOCAL_RANK", 0))


def is_distributed() -> bool:
    """True when this process is one of several ranks (torchrun launch)."""
    return get_world_size() > 1


def is_initialized() -> bool:
    """True once the process group is up (and there is more than one rank)."""
    return dist.is_available() and dist.is_initialized()


# Utility: only rank 0 prints logs
def is_main_process(rank=None):
    return (get_rank() if rank is None else rank) == 0


def dist_sync_processes():
    if not is_initialized():
        return
    # Name the device: a bare barrier() has to guess which GPU this rank owns
    # ("devices used by this process are currently unknown … can potentially
    # cause a hang if this rank to GPU mapping is incorrect"), and its guess is
    # only right as long as local_rank happens to match.
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        dist.barrier(device_ids=[get_local_rank()])
    else:
        dist.barrier()


def init_distributed(timeout_min: int = 120) -> tuple[int, int, int]:
    """Join the process group described by the environment.

    Returns ``(rank, world_size, local_rank)``.  With ``WORLD_SIZE <= 1`` it
    only reports (0, 1, 0) — nothing is initialised, so every single-GPU entry
    point can call this unconditionally.

    The default 30 min NCCL timeout is raised to two hours: after
    `init_process_group` the next collective is the first gradient all-reduce,
    and each rank first builds its dataset (a cold sharded Arrow cache) and
    loads its checkpoint, which can take far longer than 30 min on a cluster
    filesystem while the other ranks sit in the collective.
    """
    world_size, rank, local_rank = get_world_size(), get_rank(), get_local_rank()
    if world_size <= 1:
        return rank, world_size, local_rank

    if torch.cuda.is_available():
        # bind this process to its GPU *before* the first collective, so NCCL
        # does not create its communicator on cuda:0 for every local rank
        torch.cuda.set_device(local_rank)

    if not is_initialized():
        from datetime import timedelta

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=timeout_min),
        )
    return rank, world_size, local_rank


def resolve_device(device):
    """Map a configured device (``cuda:0``) onto this rank's own GPU.

    Every rank of a node sees all of the node's GPUs (slurm allocates them to
    the single task torchrun forks from), so the configs' ``device: cuda:0``
    would put the whole node on one GPU without this.
    """
    if not is_distributed():
        return device
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        return f"cuda:{get_local_rank()}"
    return device


def broadcast_module_states(modules, src: int = 0) -> None:
    """Make every rank start from rank 0's parameters and buffers.

    torch seeds its default generator from OS entropy, so identically
    configured modules are *not* identically initialised across ranks. Without
    this the ranks would average gradients of different models from step one.
    Modules that carry no parameters, and ``None`` entries, are skipped.
    """
    if not is_initialized():
        return
    for module in modules:
        if module is None:
            continue
        tensors = []
        if hasattr(module, "parameters"):
            tensors += [p.data for p in module.parameters()]
        if hasattr(module, "buffers"):
            tensors += [b.data for b in module.buffers()]
        for tensor in tensors:
            dist.broadcast(tensor, src=src)


def average_gradients(params) -> None:
    """All-reduce the mean gradient over ranks, in place.

    ``params`` must be the same ordered list on every rank (it is: the
    optimizer's parameter list, built from identically configured modules).

    A parameter that got a gradient on *some* rank only — a loss term that saw
    no matching sample there — has a zero gradient materialised so the
    collective sees the same tensors everywhere instead of dead-locking on a
    rank-dependent subset. That matches what DDP's ``find_unused_parameters``
    contributes for an unused parameter.  Parameters that got no gradient on
    *any* rank keep ``grad = None``: a materialised zero is not neutral for
    Adam, which would still apply momentum and (non-decoupled) weight decay to
    a frozen parameter and so quietly drift it away from its single-GPU value.
    Establishing which case a parameter is in costs one small collective and a
    device sync per step, next to the gradient all-reduce itself.

    Gradients are flattened per dtype into one buffer, so a step costs one
    collective per dtype rather than one per tensor.
    """
    if not is_initialized():
        return
    world_size = get_world_size()
    params = list(params)

    device = params[0].device if params else None
    if device is None:
        return
    local = torch.tensor([p.grad is not None for p in params],
                         dtype=torch.uint8, device=device)
    dist.all_reduce(local, op=dist.ReduceOp.MAX)
    any_rank_has_grad = local.tolist()

    buckets = defaultdict(list)
    for param, has_grad in zip(params, any_rank_has_grad):
        if not has_grad:
            continue  # untouched everywhere → leave grad None, as single-GPU
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        buckets[str(param.dtype)].append(param)

    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

    for dtype in sorted(buckets):  # identical iteration order on every rank
        grads = [p.grad for p in buckets[dtype]]
        flat = _flatten_dense_tensors(grads)
        dist.all_reduce(flat)
        flat /= world_size
        for grad, reduced in zip(grads, _unflatten_dense_tensors(flat, grads)):
            grad.copy_(reduced)


def all_reduce_any(flag: bool) -> bool:
    """True if *flag* holds on any rank — for agreeing on control flow.

    A rank that skips a collective (because its loss happened to carry no
    graph) hangs every other rank in it until the timeout kills the job, so
    the decision to take an optimizer step is taken jointly rather than
    per-rank.
    """
    if not is_initialized():
        return bool(flag)
    tensor = torch.tensor(
        [1 if flag else 0],
        dtype=torch.uint8,
        device=f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu",
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return bool(tensor.item())


def _scalar_all_reduce(value: float, op) -> float:
    if not is_initialized():
        return value
    tensor = torch.tensor(
        [value],
        dtype=torch.float64,
        device=f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu",
    )
    dist.all_reduce(tensor, op=op)
    return float(tensor.item())


def all_reduce_mean(value: float) -> float:
    """Mean of a python scalar across ranks (returned unchanged if alone)."""
    if not is_initialized():
        return value
    return _scalar_all_reduce(value, dist.ReduceOp.SUM) / get_world_size()


def all_reduce_sum(value: float) -> float:
    """Sum of a python scalar across ranks — sample counts, accumulated times."""
    return _scalar_all_reduce(value, dist.ReduceOp.SUM)


def all_reduce_max(value: float) -> float:
    """Max of a python scalar across ranks.

    The right reduction for wall-clock and peak-memory numbers: the job lasts
    as long as its slowest rank and needs as much memory as its hungriest one,
    while a mean would hide both.
    """
    return _scalar_all_reduce(value, dist.ReduceOp.MAX)


def all_gather_object(obj) -> list:
    """Collect *obj* from every rank (rank order); ``[obj]`` when alone.

    For the small python structures the eval reduction needs — a dict of
    per-metric (sum, count) pairs whose keys can differ per rank, so a plain
    tensor all-reduce would not line up.
    """
    if not is_initialized():
        return [obj]
    gathered = [None] * get_world_size()
    dist.all_gather_object(gathered, obj)
    return gathered


def setup_process_group(rank, world_size, master_addr="127.0.0.1", master_port="29500"):
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # make sure each process uses its own GPU
    torch.cuda.set_device(rank)


def cleanup_process_group():
    if not is_initialized():
        return
    dist.destroy_process_group()
