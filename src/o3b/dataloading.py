"""DataLoader environment setup shared by benchmark eval and method training.

Kept free of heavy / optional dependencies (unlike o3b.multiprocessing, which
pulls in sqlalchemy) so every entry point that builds a DataLoader can import it.
"""
import logging
import os

import torch

logger = logging.getLogger(__name__)


def configure_dataloader_sharing():
    """Make multi-worker DataLoaders survive a low open-file limit.

    Torch's default 'file_descriptor' sharing strategy sends one fd per shared
    tensor from the worker to the consumer process.  Our batches carry dozens of
    tensors each (rgb / depth / mask / kpts / mesh per frame object, twice for a
    pair), and with prefetch + pin_memory several batches are in flight at once,
    so the receiving process can hold thousands of fds.  Where RLIMIT_NOFILE is
    small (slurm nodes typically 1024, workstations 1M) the fd hand-off fails as

        RuntimeError: received 0 items of ancdata

    in the pin-memory thread, and the loader then dies with "Pin memory thread
    exited unexpectedly".  Two mitigations, both applied here: raise the soft
    limit to the hard limit, and switch to 'file_system' sharing, which passes
    /dev/shm names instead of fds.  Call before building a DataLoader.

    Set O3B_SHARING_STRATEGY=file_descriptor to keep torch's default (which
    leaves no /dev/shm files behind if the process is killed with SIGKILL).
    """
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            logger.info(f"raised RLIMIT_NOFILE soft limit {soft} -> {hard}")
        except (ValueError, OSError) as exc:
            logger.warning(f"could not raise RLIMIT_NOFILE from {soft} to {hard}: {exc}")

    strategy = os.environ.get("O3B_SHARING_STRATEGY", "file_system")
    if strategy in torch.multiprocessing.get_all_sharing_strategies():
        torch.multiprocessing.set_sharing_strategy(strategy)
    else:
        logger.warning(f"unknown O3B_SHARING_STRATEGY {strategy!r}, keeping "
                       f"{torch.multiprocessing.get_sharing_strategy()!r}")
