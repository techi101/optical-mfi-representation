"""
run_remaining.py
================
Runs the ENTIRE remaining pipeline unattended, start to finish.

WHY THIS EXISTS
The experiments take a few hours in total. Rather than depending on someone
(or something) starting each stage as the previous one finishes, this script
chains all of them into a single process. Start it once and walk away.

It is deliberately defensive:
  * if one stage fails, the remaining stages still run
  * every stage's output is written to results/pipeline_log.txt
  * completed stages are SKIPPED on a re-run, so if the machine is
    interrupted you can just start it again and it picks up where it left off
  * a final report says exactly what succeeded and what did not

WHAT IT RUNS, IN ORDER
  0. waits for the main comparison (train_and_evaluate.py) to finish
  1. experiments.py generalise    -- unseen OSNR levels
  2. experiments.py ablation      -- architecture sweep
  3. experiments.py symbols       -- how short a capture is enough
  4. experiments.py robustness    -- train on one channel, test on another
  5. plots.py                     -- all figures + results_summary.txt
  6. paper/make_macros.py         -- fills every number into the manuscript
  7. verify_project.py            -- independent correctness checks

Run me:
    python run_remaining.py
"""

import os
import subprocess
import sys
import time

import config

PY = sys.executable
LOG = os.path.join(config.RESULTS_DIR, "pipeline_log.txt")
STATUS = os.path.join(config.RESULTS_DIR, "pipeline_status.txt")


def log(msg):
    line = "[{}] {}".format(time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(name, args, produces=None):
    """
    Run one stage. Returns True on success.

    `produces` is the results file the stage should create. If it already
    exists the stage is skipped, which makes the whole pipeline restartable.
    """
    if produces and os.path.exists(os.path.join(config.RESULTS_DIR, produces)):
        log("SKIP  {}  (results/{} already exists)".format(name, produces))
        return True

    log("START {}".format(name))
    t0 = time.time()
    try:
        proc = subprocess.run([PY, "-u"] + args,
                              cwd=config.PROJECT_DIR,
                              capture_output=True, text=True, timeout=6 * 3600)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write("\n----- {} stdout -----\n".format(name))
            f.write(proc.stdout or "")
            if proc.returncode != 0:
                f.write("\n----- {} stderr -----\n".format(name))
                f.write(proc.stderr or "")
        dt = (time.time() - t0) / 60
        if proc.returncode == 0:
            log("DONE  {}  ({:.1f} min)".format(name, dt))
            return True
        log("FAIL  {}  (exit {}, {:.1f} min) -- see {}".format(
            name, proc.returncode, dt, LOG))
        # print the tail of stderr so the failure is visible immediately
        for ln in (proc.stderr or "").strip().split("\n")[-12:]:
            log("      | " + ln)
        return False
    except subprocess.TimeoutExpired:
        log("FAIL  {}  (timed out after 6 h)".format(name))
        return False
    except Exception as exc:                      # noqa: BLE001
        log("FAIL  {}  ({})".format(name, exc))
        return False


def wait_for_main(timeout_min=180):
    """
    Block until the main comparison has written its results file.

    We poll for the FILE rather than watching the process, because the file is
    written only at the very end of a successful run. That also means starting
    this script early is harmless.
    """
    target = os.path.join(config.RESULTS_DIR, "main_results.json")
    if os.path.exists(target):
        log("main_results.json already present -- continuing")
        return True

    log("waiting for train_and_evaluate.py to finish "
        "(polling for results/main_results.json)...")
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        if os.path.exists(target):
            time.sleep(5)          # let the file finish being written
            log("main comparison finished")
            return True
        time.sleep(20)
    log("TIMED OUT waiting for the main comparison after {} min".format(
        timeout_min))
    return False


def main():
    log("=" * 64)
    log("PIPELINE START")
    log("=" * 64)
    t_start = time.time()

    results = {}

    if not wait_for_main():
        log("Cannot continue without main_results.json. Stopping.")
        return 1
    results["main comparison"] = True

    stages = [
        ("generalisation study", ["experiments.py", "generalise"],
         "exp_generalise.json"),
        ("architecture ablation", ["experiments.py", "ablation"],
         "exp_ablation.json"),
        ("capture-length study", ["experiments.py", "symbols"],
         "exp_symbols.json"),
        ("robustness study", ["experiments.py", "robustness"],
         "exp_robustness.json"),
    ]
    for name, args, produces in stages:
        results[name] = run(name, args, produces)

    # These three are cheap and must ALWAYS re-run, because they summarise
    # whatever results happen to exist.
    results["figures + summary"] = run("figures + summary", ["plots.py"])
    results["paper numbers"] = run("paper numbers",
                                   [os.path.join("paper", "make_macros.py")])
    results["verification"] = run("verification", ["verify_project.py"])

    # ------------------------------------------------------------------
    total = (time.time() - t_start) / 60
    lines = ["", "=" * 64, "PIPELINE FINISHED in {:.1f} min".format(total),
             "=" * 64, ""]
    for k, v in results.items():
        lines.append("  {:<26s} {}".format(k, "OK" if v else "FAILED"))
    n_fail = sum(1 for v in results.values() if not v)
    lines += ["",
              "  {} of {} stages succeeded.".format(
                  len(results) - n_fail, len(results)),
              ""]
    if n_fail:
        lines.append("  Failed stages are detailed in results/pipeline_log.txt")
    else:
        lines += ["  Everything is done. Next:",
                  "    - results/results_summary.txt   all numbers for the paper",
                  "    - figures/                      all 9 figures",
                  "    - paper/paper.tex               the manuscript",
                  ""]

    text = "\n".join(lines)
    print(text, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(STATUS, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
