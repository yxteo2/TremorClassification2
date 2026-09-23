# 90 % precision per class is not reachable — and abstention reveals why

## The question

Can the reported model reach 0.90 precision on all three classes? At full
coverage, no — and the blocking reason is arithmetic rather than pessimism: a
model cannot be more accurate than its labels, and `self_consistency_gate.md`
bounds the label-noise share of errors at ~54 %, implying at most ~19 %
mislabelling on a contrast with no gold standard even post-mortem.

The legitimate route is **abstention**: answer confidently for some patients and
refer the rest. `metrics/selective.py` existed for this and had never been run
against the reported model.

## Result — 20 splits, margin rule

| coverage | n answered | precN | precPD | precET | macroP | ET answered |
|---|---|---|---|---|---|---|
| 1.0 | 81 | 0.642 | 0.649 | 0.648 | 0.646 | 10.0 |
| 0.9 | 73 | 0.666 | 0.664 | 0.677 | 0.669 | 8.8 |
| 0.8 | 65 | 0.679 | 0.687 | 0.688 | 0.685 | 7.3 |
| 0.7 | 57 | 0.702 | 0.698 | 0.722 | 0.707 | 6.5 |
| **0.6** | 49 | 0.729 | 0.719 | **0.753** | **0.734** | 5.9 |
| 0.5 | 40 | 0.765 | 0.712 | 0.600 | 0.692 | 4.6 |
| 0.4 | 32 | 0.797 | 0.705 | 0.605 | 0.728 | 3.6 |
| 0.3 | 24 | 0.813 | 0.730 | 0.490 | 0.708 | 3.0 |
| 0.2 | 16 | **0.838** | 0.644 | 0.448 | 0.674 | 2.3 |

`max_prob` behaves the same way, peaking at macroP **0.749** at 0.5 coverage.

**No class reaches 0.90 at any coverage.** The best single figure anywhere in the
sweep is precN 0.838 at 20 % coverage — where the model answers for 16 patients.

## The finding: the errors that matter cannot be abstained away

Only **precN is monotone**. precPD peaks around 0.73–0.76 and falls. **precET
rises to 0.753 at 60 % coverage and then collapses** — 0.600, 0.490, 0.448.

Abstention works by discarding the *least confident* predictions. If the
remaining errors were low-confidence, precision would keep rising. It does not,
which means **the surviving ET errors are high-confidence ones**: the model is
confidently wrong about them.

That is the same population the self-consistency gate identified. The gate found
~54 % of errors are the *consistently-wrong* kind — the model gives both of a
patient's recordings the same wrong answer. **Two independent measurements now
agree**: those errors are stable and confident, so no confidence threshold
removes them.

> **Abstention cannot fix label noise, because the model is confident on
> mislabelled patients.**

This is worth more than a coverage number. It closes the abstention route for the
right reason and corroborates the ceiling account from a second direction.

The mechanism is visible in the last column: ET answered falls 10.0 → 5.9 → 2.3.
Below ~60 % coverage precET is computed over three or fewer patients, so it is
both noisy and dominated by whichever confident errors remain.

## Predictions, scored

1. *"N reaches 0.90 at high coverage."* — **failed.** precN tops out at 0.838 at
   20 % coverage. The error was mine: I extrapolated from this project's
   N-vs-Tremor result (0.910 / 0.924) to 3-class precN. They are different
   quantities — in the 3-class problem precN is diluted by PD and ET patients
   misassigned to N, which the binary screen never sees.
2. *"PD and ET need heavy abstention; ET may not reach 0.90 at any coverage."* —
   **held for ET**, and it is worse than "may not": ET gets *further* from 0.90
   as coverage falls.
3. *"ET's curve may be non-monotone, because confidence ranking removes ET
   patients faster than the PD patients they are confused with."* — **held,
   strikingly.** The peak is at 0.6 coverage and the collapse is 0.305 deep.

## What can honestly be claimed

* **Full coverage, 3-class:** macroP 0.646, every class 0.64–0.65. This is the
  figure to quote.
* **60 % coverage:** macroP **0.734**, with all three classes at **0.72–0.75**,
  referring 40 % of patients for specialist review. A defensible clinical
  operating point and a real +0.088 macroP over full coverage.
* **Screening only (N vs Tremor):** precision **0.910 / 0.924** at full coverage
  (`frequency/characteristics.py`). **This is where 90 % genuinely exists.**
  Separating the screening claim from the PD-vs-ET claim is stronger than one
  blended number, because only the second is label-limited.
* **0.90 on all three classes:** not available at any coverage. Do not promise
  it, and do not present a low-coverage precN of 0.838 as if it generalised —
  it is 16 patients.

## Standing

* **Abstention is closed as a route to 0.90**, for a measured reason rather than
  a null: the residual errors are confident, so thresholding on confidence
  cannot remove them.
* **Adopt 60 % coverage as the reported operating point** if a triage framing is
  wanted. It is the macroP peak under the margin rule and the last coverage at
  which all three classes improve together.
* **The corroboration is the publishable part.** The gate said the errors are
  stable and confident; selective prediction independently shows they are
  un-abstainable. Two measurements, one conclusion, different instruments.
