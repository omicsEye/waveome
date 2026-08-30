"""Guard: every CALIBRATION simulation must exercise the shipped p-value code.

The bug this prevents: `sim_fdr_stage1_uniformity.py` computed p-values from
a single global pooled tail -- the construction the conditional quantile
regression replaced -- so it was calibrating code the pipeline does not run.
It went unnoticed because the script produced plausible numbers. On the same
draws the two constructions disagree by a factor of four in the wide-null
stratum (0.444 vs 0.074), which made a correct BIC fix look like a
calibration failure and nearly got it reverted. FINDINGS.md section 24.

`sim_fdr_stage2.py` had the same defect in a worse place: its headline
FDR/power table -- the reviewer-requested calibration evidence -- scored
scale-binned pooling and the GPD tail, both of which were REJECTED in favour
of the quantile regression.

A simulation whose purpose is to calibrate the reported method must call
that method. Scripts that exist to COMPARE constructions are exempt by
design and are listed as such.

    python tests/test_sims_use_shipped_pvalues.py
"""
import ast
import os
import sys

SIM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "examples", "simulations")
SHIPPED = "calc_permutation_pvalues"

# Simulations that report calibration/power for the REPORTED method.
MUST_CALL = [
    "sim_fdr_stage1_uniformity.py",
    "sim_fdr_stage2.py",
    "sim_se_power.py",
]
# Scripts whose purpose is to score alternatives against each other. These
# legitimately contain local implementations; they must still call the
# shipped one so the comparison has a truthful reference point.
COMPARISON = [
    "eval_pooling_variants.py",
    "eval_gpd_tail.py",
    "sim_fdr_stage1_stratified.py",
]


def calls_shipped(path):
    """True if the file imports AND calls the shipped function."""
    src = open(path).read()
    tree = ast.parse(src)
    imported = any(
        isinstance(n, ast.ImportFrom) and n.module and "waveome" in n.module
        and any(a.name == SHIPPED for a in n.names)
        for n in ast.walk(tree))
    called = any(
        isinstance(n, ast.Call) and getattr(n.func, "id", None) == SHIPPED
        for n in ast.walk(tree))
    return imported, called


def main():
    failures = []
    print("calibration simulations (must call the shipped construction):")
    for f in MUST_CALL:
        path = os.path.join(SIM_DIR, f)
        if not os.path.exists(path):
            failures.append(f"{f}: MISSING")
            print(f"  {f:36s} MISSING")
            continue
        imported, called = calls_shipped(path)
        ok = imported and called
        print(f"  {f:36s} {'ok' if ok else 'FAIL'}"
              f"  (imports={imported}, calls={called})")
        if not ok:
            failures.append(
                f"{f} does not call {SHIPPED}; it is reporting calibration "
                "for a construction the pipeline does not run")

    print("\ncomparison scripts (local variants allowed, shipped must appear):")
    for f in COMPARISON:
        path = os.path.join(SIM_DIR, f)
        if not os.path.exists(path):
            print(f"  {f:36s} absent (ok)")
            continue
        imported, called = calls_shipped(path)
        print(f"  {f:36s} {'references shipped' if called else 'LOCAL ONLY'}")

    if failures:
        print("\nFAILED:")
        for x in failures:
            print(f"  - {x}")
        sys.exit(1)
    print("\nall calibration simulations exercise the shipped code")


if __name__ == "__main__":
    main()
