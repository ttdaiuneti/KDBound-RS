"""Second-machine check for KDBound-RS (reviewer comment on hardware dependence).

Re-runs a fixed subset of the timing units of the paper, one unit at a time, on another machine, with the same
locked engine code (code/e1, code/g1; SHA-256 in reference/SHA256SUMS.txt) and the same data (data/*.npz).

  python run_check.py check            -> verify files and packages, run one small unit (about 1 s)
  python run_check.py run [A|B|all]    -> run the units (resumable: finished units are skipped)
  python run_check.py summary          -> compare with the Apple M4 reference, write results/summary.txt
  python run_check.py one SLUG CRIT PARAM METHOD   (internal: one unit in a fresh process)

Part A: NRS, delta = 0.10, fold 1 (the subset re-measured on an idle machine in Section 5.1 of the paper).
Part B: FRS, sigma = 0.25, fold 1, on five tables where the engines finish.
Each unit runs in its own process with one BLAS thread. A unit is stopped after TIMEOUT seconds (recorded as
'timeout') or when its memory exceeds 75% of the RAM (recorded as 'memout'). Only ASCII is printed.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_E1 = os.path.join(HERE, "code", "e1")
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "x86_results.jsonl")
REF = os.path.join(HERE, "reference", "m4_reference.jsonl")
TIMEOUT = 1800          # the paper used 900 s; the check allows a slower machine to finish the same units
WANT = {"numpy": "2.0.2", "scipy": "1.13.1", "scikit-learn": "1.6.1"}

A = [(s, "NRS", "0.10", m) for s in ("magic-telescope", "house-16h", "creditcard", "gas-drift", "jannis", "miniboone")
     for m in ("KD-PA+B3", "KD-PA")] + \
    [(s, "NRS", "0.10", m) for s in ("magic-telescope", "creditcard", "gas-drift") for m in ("FNC", "FNC+B3")] + \
    [(s, "NRS", "0.10", "HASH") for s in ("magic-telescope", "house-16h")]
B = [(s, "FRS", "0.25", m) for s in ("magic-telescope", "house-16h", "creditcard", "internet-firewall", "miniboone")
     for m in ("KD-PA+B3", "KD-PA")] + \
    [(s, "FRS", "0.25", "FA-FPR") for s in ("magic-telescope", "house-16h")]
HEAVY = [(s, "NRS", "0.10", m) for s in ("house-16h",) for m in ("FNC", "FNC+B3")]   # about 11 GB each; run last
PARTS = {"A": A + HEAVY, "B": B, "all": A + B + HEAVY}


def env():
    e = dict(os.environ)
    e.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
             PYTHONWARNINGS="ignore")
    return e


# ---------------------------------------------------------------- one unit (child process)
def one(slug, crit, param, method):
    sys.path.insert(0, CODE_E1)
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    z = np.load(os.path.join(DATA, slug + ".npz"))
    X, y = z["X"], z["y"]
    tr = next(iter(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(y)), y)))[0]
    X, y = X[tr], y[tr]
    rec = dict(dataset=slug, criterion=crit, param=param, fold=1, method=method, n_train=int(len(y)), d=int(X.shape[1]))
    t0 = time.perf_counter()
    try:
        if crit == "NRS":
            from engines_e1 import run_nrs
            R, evals, _ = run_nrs(method, X, y, float(param))
        else:
            from frs import run_frs
            R, evals, _ = run_frs(method, X, y, float(param))
        rec.update(status="ok", R=[int(a) for a in R], evals=int(evals))
    except Exception as e:  # noqa: BLE001
        rec.update(status="memout" if type(e).__name__ == "MemOut" else "error", note=f"{type(e).__name__}: {e}")
    rec["time"] = time.perf_counter() - t0
    print(json.dumps(rec), flush=True)


# ---------------------------------------------------------------- parent: run with time and memory caps
def run_unit(u):
    try:
        import psutil
        cap = 0.75 * psutil.virtual_memory().total / 2 ** 30
    except ImportError:
        psutil, cap = None, None
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "one", *u], env=env(), cwd=HERE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t0, peak, note = time.time(), 0.0, None
    while p.poll() is None:
        if psutil is not None:
            try:
                mi = psutil.Process(p.pid).memory_info()
                gb = max(mi.rss, getattr(mi, "private", 0)) / 2 ** 30
                peak = max(peak, gb)
                if gb > cap:
                    note = "memout"
            except psutil.Error:
                pass
        if time.time() - t0 > TIMEOUT:
            note = "timeout"
        if note:
            p.kill()
            break
        time.sleep(0.5)
    out, err = p.communicate()
    base = dict(zip(("dataset", "criterion", "param", "method"), u), fold=1)
    lines = [ln for ln in out.splitlines() if ln.startswith("{")]
    if note or not lines:
        r = dict(base, status=note or "error", time=time.time() - t0, note=(err or "")[-400:])
    else:
        r = json.loads(lines[-1])
    r.update(peak_mem_gb=round(peak, 2), mem_cap_gb=round(cap, 1) if cap else None, timeout_s=TIMEOUT)
    return r


def done(path):
    if not os.path.exists(path):
        return set()
    return {(r["dataset"], r["criterion"], r["param"], r["method"]) for r in map(json.loads, open(path))}


def log(msg, path):
    print(msg, flush=True)
    with open(os.path.join(os.path.dirname(path), "run_log.txt"), "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def keep_awake(on):
    """Windows: ask the system not to sleep while the units run (no effect elsewhere)."""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))
        except Exception:  # noqa: BLE001
            pass


def run(part, path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    units = PARTS[part]; fin = done(path)
    todo = [u for u in units if u not in fin]
    log(f"{len(units)} units in part {part}, {len(units) - len(todo)} already done, {len(todo)} to run", path)
    keep_awake(True)
    try:
        for i, u in enumerate(todo, 1):
            r = run_unit(u)
            with open(path, "a") as f:
                f.write(json.dumps(r) + "\n")
            log(f"[{i}/{len(todo)}] {u[0]:18s} {u[1]} {u[2]} {u[3]:9s} {r['status']:8s} "
                f"{r.get('time', 0):8.1f} s  peak {r.get('peak_mem_gb', 0):.1f} GB", path)
    finally:
        keep_awake(False)
    log("finished; now run: python run_check.py summary", path)


# ---------------------------------------------------------------- machine and file checks
def cpu_name():
    try:
        if sys.platform.startswith("win"):
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True).stdout.strip()
        for ln in open("/proc/cpuinfo"):
            if ln.startswith("model name"):
                return ln.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor()


def versions():
    import numpy, scipy, sklearn
    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__}


def machine_info():
    info = dict(cpu=cpu_name(), machine=platform.machine(), os=platform.platform(), **versions())
    try:
        import psutil
        info.update(ram_gb=round(psutil.virtual_memory().total / 2 ** 30, 1),
                    cores_physical=psutil.cpu_count(logical=False), cores_logical=psutil.cpu_count())
        f = psutil.cpu_freq()
        if f:
            info.update(cpu_mhz_max=f.max)
    except ImportError:
        info.update(note="psutil not installed")
    try:
        import numpy
        cfg = numpy.show_config(mode="dicts")
        info.update(blas=cfg["Build Dependencies"]["blas"].get("name"))
    except Exception:  # noqa: BLE001
        pass
    return info


def check():
    ok = True
    for ln in open(os.path.join(HERE, "reference", "SHA256SUMS.txt")):
        h, f = ln.split()
        got = hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()
        if got != h:
            print("CHANGED FILE:", f); ok = False
    print("files: all 10 match SHA256SUMS.txt" if ok else "files: SOME FILES DIFFER (copy the folder again)")
    v = versions()
    for k, want in WANT.items():
        flag = "ok" if v[k] == want else f"DIFFERENT (paper used {want})"
        print(f"{k:13s} {v[k]:8s} {flag}")
    print(f"python        {v['python']}")
    try:
        import psutil  # noqa: F401
    except ImportError:
        print("psutil is missing: no memory cap (pip install psutil)"); ok = False
    info = machine_info()
    os.makedirs(RES, exist_ok=True)
    json.dump(info, open(os.path.join(RES, "machine_info.json"), "w"), indent=1)
    print("machine:", info.get("cpu"), "|", info.get("ram_gb"), "GB |", info.get("os"))
    r = run_unit(("magic-telescope", "NRS", "0.10", "KD-PA+B3"))
    ref = {(x["dataset"], x["criterion"], x["param"], x["method"]): x for x in map(json.loads, open(REF))}
    rr = ref[("magic-telescope", "NRS", "0.10", "KD-PA+B3")]
    same = r.get("R") == rr["R"]
    print(f"test unit: {r['status']}, {r.get('time', 0):.2f} s (M4: {rr['time']:.2f} s), reduct "
          f"{'IDENTICAL to M4' if same else 'DIFFERENT from M4: ' + str(r.get('R'))}")
    print("CHECK PASSED: start the run" if ok and same else "CHECK FAILED: see the lines above")


# ---------------------------------------------------------------- comparison with the M4 reference
def summary():
    import statistics as st
    ref = {(x["dataset"], x["criterion"], x["param"], x["method"]): x for x in map(json.loads, open(REF))}
    new = {(x["dataset"], x["criterion"], x["param"], x["method"]): x for x in map(json.loads, open(OUT))}
    info = json.load(open(os.path.join(RES, "machine_info.json"))) if os.path.exists(
        os.path.join(RES, "machine_info.json")) else {}
    L = [f"Machine: {info.get('cpu')} | {info.get('ram_gb')} GB | {info.get('os')} | Python {info.get('python')} "
         f"numpy {info.get('numpy')} scipy {info.get('scipy')} sklearn {info.get('scikit-learn')}",
         "Reference: Apple M4 Pro (reference/m4_reference.jsonl, same script and code)", "",
         f"{'dataset':18s} {'crit':4s} {'method':9s} {'status':8s} {'time':>8s} {'M4':>8s} {'ratio':>6s}  reduct"]
    both = same = 0; ratios = []
    for u in PARTS["all"]:
        if u not in new:
            continue
        n, r = new[u], ref.get(u, {})
        okb = n["status"] == "ok" and r.get("status") == "ok"
        tag = "-"
        if okb:
            both += 1; s = n["R"] == r["R"]; same += s; tag = "same" if s else "DIFFERENT"
            if r["time"] > 0.5:
                ratios.append(n["time"] / r["time"])
        rt = f"{n['time'] / r['time']:6.2f}" if okb and r["time"] > 0 else "     -"
        L.append(f"{u[0]:18s} {u[1]:4s} {u[3]:9s} {n['status']:8s} {n.get('time', 0):8.1f} {r.get('time', 0):8.1f} "
                 f"{rt}  {tag}")
    L += ["", f"Ordered reduct identical to M4 in {same} of {both} units that finished on both machines."]
    if ratios:
        L.append(f"Time on this machine / M4 (units > 0.5 s on M4): median {st.median(ratios):.2f}, "
                 f"range {min(ratios):.2f}-{max(ratios):.2f}, {len(ratios)} units.")
    L += ["", "Speed-ups (time of the first method / time of the second), this machine vs M4:"]
    pairs = [("KD-PA", "KD-PA+B3"), ("FNC", "KD-PA+B3"), ("FNC", "FNC+B3"), ("HASH", "KD-PA+B3"),
             ("FA-FPR", "KD-PA+B3")]
    q = []
    for (s, c, p) in sorted({u[:3] for u in PARTS["all"]}):
        for a, b in pairs:
            ka, kb = (s, c, p, a), (s, c, p, b)
            if all(k in new and k in ref and new[k]["status"] == ref[k]["status"] == "ok" for k in (ka, kb)):
                x = new[ka]["time"] / new[kb]["time"]; m = ref[ka]["time"] / ref[kb]["time"]
                if ref[kb]["time"] > 0.5:
                    q.append(x / m)
                L.append(f"  {s:18s} {c} {a:7s}/{b:9s} {x:7.2f} {m:7.2f}")
    if q:
        L.append(f"Speed-up here / speed-up on M4 (denominator > 0.5 s on M4): median {st.median(q):.2f}, "
                 f"range {min(q):.2f}-{max(q):.2f}, {len(q)} pairs.")
    text = "\n".join(L)
    print(text)
    open(os.path.join(RES, "summary.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    a = sys.argv[1:] or ["help"]
    if a[0] == "one":
        one(*a[1:5])
    elif a[0] == "check":
        check()
    elif a[0] == "run":
        out = a[a.index("--out") + 1] if "--out" in a else OUT
        run(a[1] if len(a) > 1 and a[1] in PARTS else "all", out)
    elif a[0] == "summary":
        summary()
    else:
        print(__doc__)
