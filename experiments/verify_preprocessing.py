"""Absolute checks of every preprocessing stage against signals with a known answer.

The protocol's safeguards -- patient-level splits, paired bootstraps, permutation
nulls -- all compare arms, so a defect every arm shares is invisible to them. A
1 % frequency-axis stretch survived 68 reports that way. This file is the
absolute check: synthetic rotations, tones, FM tones and harmonics whose correct
descriptors are known analytically, pushed through every stage that feeds a
model.

It found the two defects fixed alongside it (`descriptor_trajectory_fix.md`):
`describe()`'s Q-factor spanned every supra-half-max bin rather than the peak,
and the IF trajectory's end points were band-pass transients. Run it after any
change to `signal_processing/`, `frequency/` or `common/cohorts.py`.

Run: ``python -m experiments.verify_preprocessing``  (exit code = number of failures)
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.signal import resample_poly, butter, filtfilt
from signal_processing.quaternion import (angular_velocity_from_quaternions, quat_multiply,
                                          _normalize_quaternions)
from signal_processing.transforms import METHODS, _band, F_MIN, F_MAX
from signal_processing.stability import stability_features, if_trajectory, trajectory_table
from signal_processing.tremor_physics import axis_features, harmonic_features
from frequency.descriptors import describe
from common.cohorts import logbin
from common.load_2025 import select_task_epoch
import experiments.final_model as FM
from common.data import Recording
from pathlib import Path

FS = 100.0; T = 15.5; t = np.arange(int(T*FS))/FS
res = []
def check(name, ok, detail=""):
    res.append((name, bool(ok))); print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)

# ---------- helpers ----------
def quat_from_body_omega(omega, fs, conv="xyzw"):
    """Integrate q_dot = 0.5 * q (x) omega_body (scalar-last)."""
    n = len(omega); q = np.zeros((n, 4)); q[0] = [0, 0, 0, 1.0]
    dt = 1/fs
    for i in range(1, n):
        w = omega[i-1]; th = np.linalg.norm(w)*dt
        if th < 1e-12: dq = np.array([0,0,0,1.0])
        else:
            ax = w/np.linalg.norm(w); dq = np.concatenate([ax*np.sin(th/2), [np.cos(th/2)]])
        q[i] = quat_multiply(q[i-1][None], dq[None], conv)[0]          # body-frame update: q * dq
        q[i] /= np.linalg.norm(q[i])
    return q
def tone(f0, A=1.0): return A*np.sin(2*np.pi*f0*t)

# ---------- (a) quaternion -> angular velocity ----------
f0, A = 6.0, 0.5                      # rad/s amplitude, tremor-sized
axis = np.array([0.3, 0.8, 0.52]); axis /= np.linalg.norm(axis)
om_body = np.outer(tone(f0, A), axis)                       # (T,3)
# analytic single-axis rotation: theta = int omega dt, q = [axis sin(th/2), cos(th/2)] (xyzw).
# An Euler-integrated sequence lags the true omega by half a sample and reads as an
# 18.7 % "frame error" at 6 Hz; that was the test's defect, not the code's.
th = -(A / (2 * np.pi * f0)) * np.cos(2 * np.pi * f0 * t)
q = np.column_stack([np.outer(np.sin(th / 2), axis), np.cos(th / 2)])
om = angular_velocity_from_quaternions(q, FS, "xyzw")       # (T-2,3)
proj = om @ axis
fpk = np.fft.rfftfreq(len(proj), 1/FS)[np.argmax(np.abs(np.fft.rfft(proj - proj.mean())))]
amp = np.sqrt(2)*proj[100:-100].std()
check("quat->omega frequency", abs(fpk-f0) < 0.1, f"peak {fpk:.3f} Hz vs {f0}")
check("quat->omega amplitude", abs(amp/A - 1) < 0.03, f"recovered {amp:.4f} vs {A} (ratio {amp/A:.4f})")
off = np.linalg.norm(om - om_body[1:-1], axis=1).max()
check("quat->omega matches analytic omega (central-difference bound 2.4 %)", off < 0.03*A, f"max |err| {off:.2e} rad/s = {100*off/A:.2f}% of A")
# antipodal flips injected
qf = q.copy(); idx = np.random.default_rng(0).choice(len(q), 40, replace=False); qf[idx] *= -1
om2 = angular_velocity_from_quaternions(qf, FS, "xyzw")
check("quat sign-flip robustness", np.abs(om2-om).max() < 1e-4, f"max diff {np.abs(om2-om).max():.2e}")

# ---------- (b) NewData path: resample 128->100 on quaternions, then omega ----------
t128 = np.arange(int(T*128))/128
th128 = -(A / (2 * np.pi * f0)) * np.cos(2 * np.pi * f0 * t128)
q128 = np.column_stack([np.outer(np.sin(th128 / 2), axis), np.cos(th128 / 2)])
q100 = resample_poly(q128, 25, 32, axis=0)
omr = angular_velocity_from_quaternions(q100, FS, "xyzw") @ axis
fpk = np.fft.rfftfreq(len(omr), 1/FS)[np.argmax(np.abs(np.fft.rfft(omr-omr.mean())))]
ampr = np.sqrt(2)*omr[100:-100].std()
check("resample 128->100 frequency", abs(fpk-f0) < 0.1, f"peak {fpk:.3f}")
check("resample 128->100 amplitude", abs(ampr/A-1) < 0.03, f"ratio {ampr/A:.4f}")

# ---------- (c) all estimators: peak location and power scaling ----------
x1 = np.atleast_2d(tone(f0, 1.0)); x2 = np.atleast_2d(tone(f0, 2.0))
for m in sorted(METHODS):
    try:
        f, P = METHODS[m](x1); f, P = np.asarray(f,float), np.asarray(P,float)
        fp = f[np.argmax(P)]; df = np.median(np.diff(f)) if len(f) > 1 else 1
        _, P2 = METHODS[m](x2); ratio = np.asarray(P2,float).max()/P.max()
        check(f"estimator {m}: peak within 1 bin", abs(fp-f0) <= df+1e-9, f"peak {fp:.3f}, bin {df:.3f}")
        check(f"estimator {m}: power scales x4 for x2 amplitude", abs(ratio-4) < 0.4, f"ratio {ratio:.2f}")
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[SKIP] estimator {m}: optional dependency missing ({e})", flush=True)
    except Exception as e:
        check(f"estimator {m}: runs", False, f"{type(e).__name__}: {e}")

# ---------- (d) band edges / GRID interpolation ----------
for fe in (3.13, 14.84):
    f, P = METHODS["multitaper"](np.atleast_2d(tone(fe)))
    v = np.interp(FM.GRID, f, P, left=0.0, right=0.0)
    check(f"GRID keeps a tone at {fe} Hz", v.max() > 0.5*P.max(), f"grid max/native max {v.max()/P.max():.2f}")

# ---------- (e) logbin ----------
e = np.linspace(0, 64, 17).round().astype(int)
check("logbin edges partition 0..64 exactly", e[0]==0 and e[-1]==64 and np.all(np.diff(e)>0), f"edges {e.tolist()}")
X = np.exp(np.arange(64, dtype=float))[None]
lb = logbin(X); check("logbin = mean of log within bin", np.allclose(lb[0], [np.mean(np.arange(64)[e[i]:e[i+1]]) for i in range(16)], atol=1e-6))

# ---------- (f) describe ----------
f, P = METHODS["welch"](np.atleast_2d(tone(f0)))
d1 = describe(f, P); _, P2 = METHODS["welch"](np.atleast_2d(tone(f0, 2.0))); d2 = describe(f, P2)
check("describe: max/mean/median freq on a tone", max(abs(d1["max_freq"]-f0), abs(d1["mean_freq"]-f0), abs(d1["median_freq"]-f0)) < 0.5,
      f"max {d1['max_freq']:.2f} mean {d1['mean_freq']:.2f} median {d1['median_freq']:.2f}")
check("describe: total_power +0.602 for x2 amplitude", abs((d2["total_power"]-d1["total_power"])-0.602) < 0.05, f"delta {d2['total_power']-d1['total_power']:.3f}")
# Q-factor with a second supra-half-max component (harmonic at 2f0, 0.8x amplitude)
xh = np.atleast_2d(tone(f0) + 0.8*tone(2*f0))
fh, Ph = METHODS["welch"](xh); dh = describe(fh, Ph)
pk = int(np.argmax(Ph)); half = Ph[pk]/2
# contiguous half-power width around the main peak
lo = pk
while lo > 0 and Ph[lo-1] >= half: lo -= 1
hi = pk
while hi < len(Ph)-1 and Ph[hi+1] >= half: hi += 1
q_contig = fh[pk]/(fh[hi]-fh[lo]+1e-12)
check("describe: Q uses the CONTIGUOUS half-power width of the peak", abs(dh["q_factor"]-q_contig)/q_contig < 0.2,
      f"describe Q {dh['q_factor']:.2f} vs contiguous-peak Q {q_contig:.2f} (single tone Q {d1['q_factor']:.2f})")

# ---------- (g) stability features ----------
st_stable = stability_features(tone(f0), FS)
fm = np.sin(2*np.pi*(f0*t + (0.5/(2*np.pi*0.3))*np.sin(2*np.pi*0.3*t)))   # 6 +/- 0.5 Hz at 0.3 Hz
st_fm = stability_features(fm, FS)
check("TSI ~0 on a stable tone", st_stable["tsi"] < 0.05, f"tsi {st_stable['tsi']:.4f} if_std {st_stable['if_std']:.4f}")
check("if_std reads FM deviation (0.5 Hz peak -> ~0.35 rms)", abs(st_fm["if_std"]-0.354) < 0.08, f"if_std {st_fm['if_std']:.3f}")

# ---------- (h) IF trajectory ----------
tr_s = if_trajectory(tone(f0), FS); tr_f = if_trajectory(fm, FS)
check("IF trajectory flat on a stable tone", np.abs(tr_s[0]).max() < 0.1, f"max |IF dev| {np.abs(tr_s[0]).max():.3f}")
check("IF trajectory swings ~+/-0.5 Hz on FM tone", 0.3 < np.abs(tr_f[0]).max() < 0.7, f"max |IF dev| {np.abs(tr_f[0]).max():.3f}")
# 3-axis mean mode on a single-axis oscillation must not zero out
sig3 = np.outer(axis, fm) + 0.01*np.random.default_rng(1).standard_normal((3, len(t)))
rec = [Recording(x=sig3.astype(np.float32), y=1, subject="s1", path=Path("x"))]
tt = trajectory_table(rec, ch=slice(0,3), fs=FS)[0][0]
check("trajectory_table mean-axis keeps FM on a single-axis oscillation", np.abs(tt[0]).max() > 0.15, f"max |IF dev| {np.abs(tt[0]).max():.3f} (1-axis {np.abs(tr_f[0]).max():.3f})")

# ---------- (i) axis features: rotation invariance ----------
ax = axis_features(sig3, FS)
Rr = np.linalg.qr(np.random.default_rng(2).standard_normal((3,3)))[0]
ax2 = axis_features(Rr @ sig3, FS)
check("axis_features: linearity ~1 for a single-axis oscillation", ax["linearity"] > 0.95, f"linearity {ax['linearity']:.4f}")
check("axis_features: rotation-invariant", max(abs(ax[k]-ax2[k]) for k in ax) < 1e-6, f"max diff {max(abs(ax[k]-ax2[k]) for k in ax):.1e}")

# ---------- (j) harmonics ----------
hf = harmonic_features(np.atleast_2d(tone(f0) + 0.5*tone(2*f0)), FS)
check("harmonic_features: h2_ratio ~ 0.25 for a 0.5-amplitude 2nd harmonic", abs(hf["h2_ratio"]-0.25) < 0.06, f"h2 {hf['h2_ratio']:.3f}")

# ---------- (k) epoch selection ----------
xe = np.zeros((9, int(38*FS))); xe[3:6, 1500:2500] = np.outer(axis, tone(f0)[:1000])
ep = select_task_epoch(xe, fs=FS, win_s=10.0)
check("select_task_epoch returns exactly 10 s", ep.shape[1] == 1000, f"{ep.shape[1]} samples")
check("select_task_epoch lands on the tremor window", np.abs(ep[3:6]).max() > 0.1)

n_fail = sum(not ok for _, ok in res)
print(f"\n{len(res)} checks, {n_fail} failed")
print("MARKER_DONE", flush=True)
raise SystemExit(n_fail)
