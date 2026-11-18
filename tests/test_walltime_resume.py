"""A job that continues itself at the time limit (platform key `resubmit_on_walltime`).

Two halves: the job script (`cli._make_sbatch_script`) traps slurm's SIGUSR1 and
submits itself again; `o3b.run` sends the continuation back to the run's checkpoint.
"""
import json
import os
import signal
import subprocess
import time

import pytest

from o3b import run as o3b_run
from o3b.cli import _make_sbatch_script
from o3b.method.method import Method, register_method


@register_method("_ResumableTestMethod")
class _Resumable(Method):
    resume_checkpoint_name = "m.ckpt"


@register_method("_NotResumableTestMethod")
class _NotResumable(Method):
    pass


def _env(monkeypatch, path, attempt):
    monkeypatch.setenv("O3B_RESUME_FILE", str(path))
    monkeypatch.setenv("O3B_RESUME_ATTEMPT", str(attempt))


def test_first_job_records_and_continuation_returns(tmp_path, monkeypatch):
    state = tmp_path / "job.resume.json"
    first = tmp_path / "1117_101010__bench__abl"
    first.mkdir()
    (first / "m.ckpt").write_bytes(b"x")

    _env(monkeypatch, state, 0)
    method, train = {"class_name": "_ResumableTestMethod"}, {"checkpoint_dir": str(first)}
    o3b_run._resume_apply("1117_101010__bench__abl", method, train, True)
    assert json.loads(state.read_text()) == {"bench__abl": {"checkpoint_dir": str(first)}}
    assert "checkpoint" not in method                      # the first job starts as configured

    # the continuation has a new timestamp and so a new default directory
    _env(monkeypatch, state, 1)
    method = {"class_name": "_ResumableTestMethod", "checkpoint": "/an/older/resume.ckpt"}
    train = {"checkpoint_dir": str(tmp_path / "1117_221010__bench__abl")}
    o3b_run._resume_apply("1117_221010__bench__abl", method, train, True)
    assert train["checkpoint_dir"] == str(first)
    assert method["checkpoint"] == str(first / "m.ckpt")


def test_continuation_without_a_checkpoint_keeps_the_config(tmp_path, monkeypatch):
    state = tmp_path / "job.resume.json"
    first = tmp_path / "a__bench"
    first.mkdir()                                          # killed before the first save
    state.write_text(json.dumps({"bench": {"checkpoint_dir": str(first)}}))
    _env(monkeypatch, state, 1)
    method, train = {"class_name": "_ResumableTestMethod", "checkpoint": "/cfg.ckpt"}, {"checkpoint_dir": "/new"}
    o3b_run._resume_apply("b__bench", method, train, True)
    assert train["checkpoint_dir"] == str(first) and method["checkpoint"] == "/cfg.ckpt"


def test_method_that_cannot_resume_registers_nothing(tmp_path, monkeypatch):
    state = tmp_path / "job.resume.json"
    _env(monkeypatch, state, 0)
    o3b_run._resume_apply("a__bench", {"class_name": "_NotResumableTestMethod"}, {"checkpoint_dir": "/d"}, True)
    assert not state.exists()                              # so the job script does not continue it


def test_no_env_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("O3B_RESUME_FILE", raising=False)
    method, train = {"class_name": "_ResumableTestMethod"}, {"checkpoint_dir": "/d"}
    o3b_run._resume_apply("a__bench", method, train, True)
    assert method == {"class_name": "_ResumableTestMethod"} and train == {"checkpoint_dir": "/d"}


def test_finished_run_is_skipped_in_a_continuation(tmp_path, monkeypatch):
    state = tmp_path / "job.resume.json"
    calls = []
    monkeypatch.setattr(o3b_run, "_run_bench_run_with_cfg_impl", lambda raw, name: calls.append(name))
    _env(monkeypatch, state, 0)
    o3b_run._run_bench_run_with_cfg({}, "t0__bench__a")
    assert json.loads(state.read_text())["bench__a"]["done"] is True
    _env(monkeypatch, state, 1)
    o3b_run._run_bench_run_with_cfg({}, "t1__bench__a")    # finished by the first job
    o3b_run._run_bench_run_with_cfg({}, "t1__bench__b")    # not reached by it
    assert calls == ["t0__bench__a", "t1__bench__b"]


# ── the job script ────────────────────────────────────────────────────────────

def _script(tmp_path, **cfg):
    cfg = dict(walltime="00:10:00", path_home=str(tmp_path), exclusive_nodes=True,
               resubmit_on_gpu_error=False, **cfg)
    self_path = tmp_path / "job.sh"
    self_path.write_text(_make_sbatch_script(cfg, "job", {"A": "b"}, str(tmp_path / "run.sh"),
                                             self_path=str(self_path)))
    return self_path


def test_off_by_default_leaves_the_script_alone(tmp_path):
    text = _script(tmp_path).read_text()
    assert "USR1" not in text.split("#SBATCH --signal")[1].split("\n", 1)[1]
    assert text.rstrip().endswith(f"bash {tmp_path / 'run.sh'}")


def _run_with_signal(tmp_path, script, args=(), register=True, sbatch_ok=True, restart=0):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    calls = tmp_path / "calls.log"
    (bin_dir / "sbatch").write_text(
        f'#!/bin/bash\necho "sbatch $*" >> {calls}\n' + ("echo 4242\n" if sbatch_ok else "echo denied; exit 1\n"))
    (bin_dir / "scontrol").write_text(f'#!/bin/bash\necho "scontrol $*" >> {calls}\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    reg = 'echo \'{"k": {"checkpoint_dir": "/x"}}\' > "$O3B_RESUME_FILE"\n' if register else ""
    (tmp_path / "run.sh").write_text(
        '#!/bin/bash\necho "run attempt=$O3B_RESUME_ATTEMPT"\n' + reg + "sleep 4\nexit 3\n")
    (tmp_path / "slurm_jobs").mkdir(exist_ok=True)
    log = tmp_path / "slurm_jobs" / "job_77.o"
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "SLURM_JOB_ID": "77",
           "SLURM_RESTART_COUNT": str(restart)}
    with open(log, "a") as out:
        proc = subprocess.Popen(["bash", str(script), *args], env=env, stdout=out, stderr=out)
        time.sleep(1.5)
        proc.send_signal(signal.SIGUSR1)
        rc = proc.wait(timeout=30)
    return rc, (calls.read_text() if calls.exists() else ""), log.read_text()


def test_script_continues_itself_and_outlives_the_signal(tmp_path):
    script = _script(tmp_path, resubmit_on_walltime=True, resubmit_walltime_max=2)
    rc, calls, log = _run_with_signal(tmp_path, script)
    assert rc == 3                                         # the run's own exit code, after the trap
    assert calls.strip() == f"sbatch --parsable --dependency=afterany:77 {script} 1  1"
    assert "continuing as job 4242 (continuation 1/2)" in log and "run attempt=0" in log


def test_script_stops_at_the_cap_and_without_a_checkpoint(tmp_path):
    script = _script(tmp_path, resubmit_on_walltime=True, resubmit_walltime_max=2)
    rc, calls, log = _run_with_signal(tmp_path, script, register=False)
    assert calls == "" and "no run of this job registered" in log
    rc, calls, log = _run_with_signal(tmp_path, script, args=("1", "", "2"))
    assert calls == "" and "all 2 continuations used" in log and "run attempt=2" in log


def test_script_requeues_where_sbatch_is_refused(tmp_path):
    script = _script(tmp_path, resubmit_on_walltime=True, resubmit_walltime_max=2)
    rc, calls, log = _run_with_signal(tmp_path, script, sbatch_ok=False)
    assert calls.strip().splitlines()[-1] == "scontrol requeue 77"
    # the requeued job has the same id, arguments and log: counted by the marker
    rc, calls, log = _run_with_signal(tmp_path, script, sbatch_ok=False, restart=1)
    assert "run attempt=1" in log and "(continuation 2/2)" in log
