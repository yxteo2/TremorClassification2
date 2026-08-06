"""Stage 1+2 benchmark: which signal-processing method best discriminates tremor,
and which deep architecture exploits it best.

Self-contained folder. Stage 1 (`benchmark.py`) ranks time-frequency methods by
how well simple, interpretable frequency descriptors (max/mean/median frequency,
etc.) separate N-vs-Tremor and PD-vs-ET. Stage 2 (`deep.py`) trains deep models
on the top methods and compares architectures.
"""
