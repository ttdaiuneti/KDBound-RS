"""E1 timing runner (e1_prereg.md). Resumable, append-only; finished datasets are locked and never re-run.

python run_e1.py [slug ...]                 -> run the grid for the given (default: all kept) datasets
python run_e1.py one SLUG CRIT PARAM FOLD METHOD   -> one run in a fresh process (used internally)

Unit of work = (dataset, criterion, param, fold, method): reduction on the training part of that fold
(StratifiedKFold(5, shuffle=True, random_state=0), the eval_harness split). Each unit runs in its own process with
one BLAS thread and a 900 s alarm; WORKERS units run concurrently (user decision 2026-09-29); FNC-family units
run one at a time (memory). Amendment 2026-09-30 (e1_prereg.md section 10): every HEAVY method (FNC family and the
pair-enumerating engines + B3) runs one at a time, and each unit is killed and recorded as 'memout' when its phys_footprint exceeds
MEM_CAP_GB or the system memory pressure turns critical. Skip rule: a (dataset, criterion, param, method) that times out or runs out of memory
on fold 1 is recorded as 'skipped-after-fold1' for folds 2-5 without running.
Outputs: e1_raw.jsonl (one line per unit), results/reducts/<slug>__<crit>__<param>__fold<k>__<method>.json,
e1_locked.txt, run_e1.log.
"""
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "e1_raw.jsonl"); LOCK = os.path.join(HERE, "e1_locked.txt")
RED = os.path.join(HERE, "results", "reducts")
LIMIT, WORKERS = 900, 4
DELTAS = (0.05, 0.10, 0.15, 0.20)
SIGMAS = ("0.25", "std")
ABLATION_DELTA = "0.10"
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
           VECLIB_MAXIMUM_THREADS="1", PYTHONWARNINGS="ignore")
_io = threading.Lock(); _heavy = threading.Semaphore(1)
# section 10 (2026-09-30): two machine crashes (Jetsam + watchdog panic) with four pair-enumerating +B3 units at once
HEAVY = {"HASH+B3", "CABIN+B3", "BUCKET+B3", "BF-PA+B3"}
MEM_CAP_GB = 32


def is_heavy(method):
    return method.startswith("FNC") or method in HEAVY


def footprint_gb(pid):
    """macOS phys_footprint (resident + compressed; what Activity Monitor shows) via proc_pid_rusage.
    RSS alone misses compressed pages: CABIN+B3 on run-or-walk showed RSS 6.5 GB at footprint 89 GB."""
    import ctypes
    class RU(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f, ctypes.c_uint64) for f in (
            "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size",
            "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime")]
    ru = RU()
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    if lib.proc_pid_rusage(int(pid), 0, ctypes.byref(ru)) != 0:
        return 0.0
    return ru.phys_footprint / 2 ** 30


def run_capped(cmd):
    """Run one unit; kill it on phys_footprint (incl. compressed) > MEM_CAP_GB or critical system memory pressure. Returns (stdout, stderr, note)."""
    p = subprocess.Popen(cmd, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=HERE)
    out, note, t0, peak = {}, None, time.time(), 0.0

    def drain(name, stream):
        out[name] = stream.read()
    th = [threading.Thread(target=drain, args=(k, getattr(p, k)), daemon=True) for k in ("stdout", "stderr")]
    for t in th:
        t.start()
    while p.poll() is None:
        rss = footprint_gb(p.pid); peak = max(peak, rss)
        pressure = int(subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                                      capture_output=True, text=True).stdout.strip() or 1)
        if rss > MEM_CAP_GB or pressure >= 4:
            note = f"killed: footprint={rss:.1f} GB, pressure={pressure}"
        elif time.time() - t0 > LIMIT + 120:
            note = "hard kill"
        if note:
            p.kill(); p.wait(); break
        time.sleep(2)
    for t in th:
        t.join()
    return out.get("stdout", ""), out.get("stderr", ""), note, time.time() - t0, peak


def code_version():
    h = hashlib.sha256()
    for f in ("engines_e1.py", "frs.py", "fara.py", "data_e1.py", "run_e1.py", "../g1/methods.py"):
        h.update(open(os.path.join(HERE, f), "rb").read())
    return h.hexdigest()[:12]


def kept(crit):
    """Per-criterion degeneracy rule (user 2026-09-29): |R| >= 2 on the capped full table (NRS delta 0.1, FRS sigma 0.25)."""
    f = {"NRS": "degeneracy_screen.log", "FRS": "degeneracy_screen_frs.log"}[crit]
    return {ln.split()[0] for ln in open(os.path.join(HERE, f)) if "KEEP" in ln}


def grid(slug):
    from engines_e1 import NRS_METHODS
    from frs import METHODS as FRS_METHODS
    # lean grid (amendment 2026-09-29, e1_prereg.md section 8): main methods at every delta, the engine ablation
    # (every engine +/- B3) at delta = 0.10 only; FRS without BF-DAR (FA-FPR = BF + positive approximation)
    main_nrs = ("KD-PA", "KD-PA+B3", "FNC", "FNC+B3", "HASH", "FARA")
    g = [("NRS", f"{d:.2f}", m) for d in DELTAS for m in NRS_METHODS
         if m in main_nrs or f"{d:.2f}" == ABLATION_DELTA] if slug in kept("NRS") else []
    g += [("FRS", s, m) for s in SIGMAS for m in FRS_METHODS if not m.startswith("BF-DAR")] \
        if slug in kept("FRS") else []
    return g


def folds(y):
    from sklearn.model_selection import StratifiedKFold
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(y)), y))


def one(slug, crit, param, fold, method):
    from data_e1 import load
    X, y = load(slug); tr = folds(y)[fold - 1][0]; X, y = X[tr], y[tr]

    def alarm(*_):
        raise TimeoutError
    signal.signal(signal.SIGALRM, alarm); signal.alarm(LIMIT)
    rec = dict(dataset=slug, criterion=crit, param=param, fold=fold, method=method, n_train=int(len(y)),
               d=int(X.shape[1]), code=code_version())
    t0 = time.perf_counter()
    try:
        if crit == "NRS":
            from engines_e1 import run_nrs
            R, evals, steps = run_nrs(method, X, y, float(param))
        else:
            from frs import run_frs
            R, evals, steps = run_frs(method, X, y, param if param == "std" else float(param))
        rec.update(status="ok", R=[int(a) for a in R], evals=int(evals))
    except TimeoutError:
        rec.update(status="timeout"); steps = None
    except Exception as e:  # noqa: BLE001  (MemOut and genuine errors are both recorded, never retried silently)
        rec.update(status="memout" if type(e).__name__ == "MemOut" else "error", note=f"{type(e).__name__}: {e}")
        steps = None
    signal.alarm(0)
    rec["time"] = time.perf_counter() - t0
    rec["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    if steps is not None and not os.environ.get("E1_NO_REDUCT"):    # idle re-measure must not overwrite E1 reducts
        os.makedirs(RED, exist_ok=True)
        with open(os.path.join(RED, f"{slug}__{crit}__{param}__fold{fold}__{method}.json"), "w") as f:
            json.dump(dict(rec, steps=steps), f, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(rec), flush=True)


SUPER = os.path.join(HERE, "e1_superseded.jsonl")


def superseded():
    """Rows replaced after a fix (e1_prereg.md sections 9-10): key (dataset, criterion, param, fold, method, code) ->
    set of row times, or None = every row with that key. 'time' pins an exact row when a re-measured row shares the
    code hash (miniboone, 2026-10-01)."""
    out = {}
    if not os.path.exists(SUPER):
        return out
    for r in map(json.loads, open(SUPER)):
        k = (r["dataset"], r["criterion"], r["param"], r["fold"], r["method"], r.get("code"))
        if "time" in r and out.get(k, set()) is not None:
            out.setdefault(k, set()).add(r["time"])
        else:
            out[k] = None
    return out


def is_superseded(r, sup):
    k = (r["dataset"], r["criterion"], r["param"], r["fold"], r["method"], r.get("code"))
    if k not in sup:
        return False
    return sup[k] is None or r.get("time") in sup[k]


def done_keys():
    if not os.path.exists(OUT):
        return {}
    sup = superseded()
    return {(r["dataset"], r["criterion"], r["param"], r["fold"], r["method"]): r["status"]
            for r in map(json.loads, open(OUT)) if not is_superseded(r, sup)}


def launch(slug, crit, param, fold, method):
    key = (slug, crit, param, fold, method)
    base = dict(zip(("dataset", "criterion", "param", "fold", "method"), key))
    sem = _heavy if is_heavy(method) else None
    if sem:
        sem.acquire()
    try:
        stdout, stderr, note, wall, peak = run_capped([sys.executable, __file__, "one", slug, crit, param, str(fold), method])
        lines = [ln for ln in stdout.splitlines() if ln.startswith("{")]
        if note == "hard kill":
            line = json.dumps(dict(base, status="timeout", time=LIMIT, note=note))
        elif note:
            line = json.dumps(dict(base, status="memout", time=wall, note=note, code=code_version()))
        else:
            line = lines[-1] if lines else json.dumps(dict(base, status="error", stderr=stderr[-800:]))
        line = json.dumps(dict(json.loads(line), peak_footprint_gb=round(peak, 2)))  # sampled every 2 s
    finally:
        if sem:
            sem.release()
    r = json.loads(line)
    with _io:
        with open(OUT, "a") as f:
            f.write(line + "\n")
        print(f"{time.strftime('%H:%M:%S')} {slug} {crit} {param} f{fold} {method:10s} {r['status']:8s} "
              f"t={r.get('time', float('nan')):7.1f}s |R|={len(r.get('R', []))} evals={r.get('evals', '-')}", flush=True)
    return r["status"]


def screened(slug):
    """True once both degeneracy logs contain this dataset (the FRS screen may still be running for others)."""
    return all(any(ln.split() and ln.split()[0] == slug for ln in open(os.path.join(HERE, f)))
               for f in ("degeneracy_screen.log", "degeneracy_screen_frs.log")
               if os.path.exists(os.path.join(HERE, f)))


def run_units(units):
    """Section 10: HEAVY units on one thread, the rest on WORKERS - 1 threads (so HEAVY waits never block light units)."""
    heavy = [u for u in units if is_heavy(u[-1])]; light = [u for u in units if not is_heavy(u[-1])]
    with ThreadPoolExecutor(1) as hx, ThreadPoolExecutor(WORKERS - 1) as lx:
        fs = [hx.submit(launch, *u) for u in heavy] + [lx.submit(launch, *u) for u in light]
        for f in fs:
            f.result()


def run_dataset(slug, lock=True):
    from data_e1 import load
    while not screened(slug):                                         # wait for this dataset's own screen line only
        time.sleep(60)
    load(slug)                                                        # build the capped sample once
    g = grid(slug); done = done_keys()
    run_units([(slug, u[0], u[1], 1, u[2]) for u in g if (slug, u[0], u[1], 1, u[2]) not in done])  # phase A: fold 1
    done = done_keys(); todo = []
    for crit, param, m in g:
        st1 = done.get((slug, crit, param, 1, m))
        for k in range(2, 6):
            if (slug, crit, param, k, m) in done:
                continue
            if st1 in ("timeout", "memout"):
                with _io, open(OUT, "a") as f:
                    f.write(json.dumps(dict(dataset=slug, criterion=crit, param=param, fold=k, method=m,
                                            status="skipped-after-fold1", fold1_status=st1,
                                            code=code_version())) + "\n")
            else:
                todo.append((crit, param, k, m))
    run_units([(slug,) + u for u in todo])                            # phase B: folds 2-5
    if lock:
        with _io, open(LOCK, "a") as f:
            f.write(f"{slug}\t{time.strftime('%Y-%m-%d %H:%M')}\t{code_version()}\n")


def rerun_method(method, slugs):
    """Re-run one method on already locked datasets after a fix (section 9); other methods untouched."""
    for slug in slugs:
        g = [u for u in grid(slug) if u[2] == method]; done = done_keys()
        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(lambda u: launch(slug, u[0], u[1], 1, u[2]),
                        [u for u in g if (slug, u[0], u[1], 1, u[2]) not in done]))
        done = done_keys(); todo = []
        for crit, param, m in g:
            st1 = done.get((slug, crit, param, 1, m))
            for k in range(2, 6):
                if (slug, crit, param, k, m) in done:
                    continue
                if st1 in ("timeout", "memout"):
                    with _io, open(OUT, "a") as f:
                        f.write(json.dumps(dict(dataset=slug, criterion=crit, param=param, fold=k, method=m,
                                                status="skipped-after-fold1", fold1_status=st1,
                                                code=code_version())) + "\n")
                else:
                    todo.append((crit, param, k, m))
        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(lambda u: launch(slug, *u), todo))
        print(f"=== {slug} {method} re-run done {time.strftime('%H:%M:%S')}", flush=True)


def main(slugs):
    locked = {ln.split("\t")[0] for ln in open(LOCK)} if os.path.exists(LOCK) else set()
    for s in slugs:
        if s in locked:
            print(f"{s}: locked, skipped", flush=True); continue
        print(f"=== {s} start {time.strftime('%H:%M:%S')}", flush=True)
        run_dataset(s)
        print(f"=== {s} locked {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        one(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6])
    elif len(sys.argv) > 1 and sys.argv[1] == "remeasure":         # section 10: whole locked dataset, rows superseded
        for s in sys.argv[2:]:
            print(f"=== {s} remeasure start {time.strftime('%H:%M:%S')}", flush=True)
            run_dataset(s, lock=False)
            print(f"=== {s} remeasure done {time.strftime('%H:%M:%S')}", flush=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "rerun":
        rerun_method(sys.argv[2], sys.argv[3:])
    else:
        from data_e1 import DATASETS
        from data_e1 import DATASETS as ALL
        main(sys.argv[1:] or [s for s in ALL if s in kept("NRS") or not screened(s) or s in kept("FRS")])
