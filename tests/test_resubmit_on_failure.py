"""`resubmit_on_failure`: the job script requeues itself after any non-zero exit."""
import os
import subprocess

from o3b.cli import _make_sbatch_script


def _run(tmp_path, state="RUNNING", restart=0, rc=3, out="boom", **cfg):
    cfg = dict(walltime="00:10:00", path_home=str(tmp_path), exclusive_nodes=True,
               resubmit_on_gpu_error=False, resubmit_max_attempts=3, **cfg)
    script = tmp_path / "job.sh"
    script.write_text(_make_sbatch_script(cfg, "job", {"A": "b"}, str(tmp_path / "run.sh"),
                                          self_path=str(script)))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    calls = tmp_path / "calls.log"
    (bin_dir / "scontrol").write_text(
        '#!/bin/bash\n'
        f'if [ "$1" = show ] && [ "$2" = job ]; then echo "JobId=77 JobState={state} ExcNodeList=(null)"; exit 0; fi\n'
        'if [ "$1" = show ]; then echo node1; exit 0; fi\n'
        f'echo "scontrol $*" >> {calls}\n')
    (bin_dir / "scontrol").chmod(0o755)
    (tmp_path / "run.sh").write_text(f'#!/bin/bash\necho "{out}"\nexit {rc}\n')
    (tmp_path / "slurm_jobs").mkdir(exist_ok=True)
    log = tmp_path / "slurm_jobs" / "job_77.o"
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "SLURM_JOB_ID": "77",
           "SLURM_RESTART_COUNT": str(restart), "SLURM_JOB_NODELIST": "node1"}
    with open(log, "a") as o:
        code = subprocess.run(["bash", str(script)], env=env, stdout=o, stderr=o, timeout=120).returncode
    return code, (calls.read_text() if calls.exists() else ""), log.read_text()


def test_any_failure_is_requeued_without_excluding_the_node(tmp_path):
    code, calls, log = _run(tmp_path, resubmit_on_failure=True)
    assert code == 3
    assert calls.strip() == "scontrol requeue 77"          # no ExcNodeList update
    assert "failure (exit 3)" in log and "attempt 2/3" in log


def test_fault_pattern_still_excludes_the_node(tmp_path):
    code, calls, log = _run(tmp_path, resubmit_on_failure=True, out="uncorrectable ECC error")
    assert "scontrol requeue 77" in calls and "ExcNodeList=node1" in calls
    assert "node fault on node1" in log


def test_success_cap_and_cancel_do_not_requeue(tmp_path):
    assert _run(tmp_path, rc=0, resubmit_on_failure=True)[1] == ""
    assert _run(tmp_path, restart=2, resubmit_on_failure=True)[1] == ""        # third attempt of 3
    assert _run(tmp_path, state="CANCELLED", resubmit_on_failure=True)[1] == ""


def test_off_by_default(tmp_path):
    code, calls, log = _run(tmp_path)
    assert code == 3 and calls == ""
