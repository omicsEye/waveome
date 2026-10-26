"""The simulation harness records per-stage resources and its provenance.

Each cell's pickle carries a per-stage table of wall time, CPU time and peak
memory (R1.M7 scaling curves come from the sweep), and a provenance stamp
naming the code that produced it. Checked here:

  1. CPU time and memory of a child process -- standing in for a Ray worker --
     are attributed to the stage it ran in, and an idle stage reads ~0 CPU;
  2. the provenance stamp names the commit, the package versions, and a hash
     of the installed library that matches this checkout's.

No pytest in this environment, so this runs directly:

    python tests/test_sim_stage_resources.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples", "simulations"))
import sim_waveome_hpc_run as harness  # noqa: E402

CHILD = """
import time
x = bytearray(300 * 2**20)
for i in range(0, len(x), 4096):
    x[i] = 1
t = time.time()
while time.time() - t < 1.5:
    pass
"""


def main():
    res = harness.StageResources(interval=0.2)
    subprocess.run([sys.executable, "-c", CHILD], check=True)
    res.lap("child")
    time.sleep(1.0)
    res.lap("idle")
    table = res.close().set_index("stage")
    print(table.round(3).to_string())

    child, idle = table.loc["child"], table.loc["idle"]
    assert child.cpu_s >= 1.0, f"child CPU not counted: {child.cpu_s:.2f}s"
    assert child.peak_rss_gb >= idle.peak_rss_gb + 0.25, \
        "child memory not counted"
    assert idle.cpu_s < 0.5, f"idle stage charged {idle.cpu_s:.2f}s CPU"
    assert 0.9 <= idle.wall_s <= 2.0, idle.wall_s
    print("[ok] child process's CPU and memory land in its own stage")

    prov = harness.provenance()
    print({k: v for k, v in prov.items() if k != "packages"})
    assert re.fullmatch(r"[0-9a-f]{40}", prov["git_commit"] or ""), prov["git_commit"]
    assert prov["packages"]["gpflow"] and prov["packages"]["tensorflow"]
    assert prov["waveome_sha256"] is not None
    assert prov["waveome_sha256"] == prov["checkout_waveome_sha256"], \
        "installed waveome differs from the checkout (reinstall before testing)"
    print("[ok] provenance stamp: commit, versions, installed library matches checkout")

    # A stale install (one file differs) must hash differently
    pkg = os.path.join(ROOT, "waveome")
    rel = sorted(f for f in os.listdir(pkg) if f.endswith(".py"))
    with tempfile.TemporaryDirectory() as stale:
        for f in rel:
            shutil.copy(os.path.join(pkg, f), stale)
        assert harness._tree_sha256(stale, rel) == harness._tree_sha256(pkg, rel)
        with open(os.path.join(stale, "model_search.py"), "a") as handle:
            handle.write("\n# edited after install\n")
        assert harness._tree_sha256(stale, rel) != harness._tree_sha256(pkg, rel)
    print("[ok] a library edited after install is detected")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
