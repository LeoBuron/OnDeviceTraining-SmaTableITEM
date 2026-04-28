"""Conservative static stack bound for stage1_pretrain from -fstack-usage data.

One-off (not per-trial):
    rm -rf build/HOST-Debug
    cmake --preset HOST-Debug -DODT_EXAMPLE=stage1_pretrain -DCMAKE_C_FLAGS=-fstack-usage
    cmake --build --preset HOST-Debug --target HOST
    uv run tools/stack_bound.py --build-dir build/HOST-Debug

Method: sum the largest frame per function along the hard-coded call chains,
plus the worst layer forward/backward/kernel leaf (vtable expanded by regex),
plus a fixed libc margin. The train and eval chains do NOT nest — both are
called from trainMain — so the bound takes the deeper of the two:
    bound = sum(SHARED) + max(sum(TRAIN_CHAIN), sum(EVAL_CHAIN))
            + worst leaf + libc margin
Limitations (documented, accepted): the chains are maintained by hand from
ODT's known structure; no recursion and no indirect calls other than the
layer vtable exist on these paths.
"""
import argparse
import re
import sys
from pathlib import Path

SHARED = ["main", "trainMain"]
TRAIN_CHAIN = ["trainingEpochDefault", "trainingBatchDefault",
               "calculateGradsSequential", "calculateGradsImpl"]
EVAL_CHAIN = ["evaluationEpochWithMetrics", "evaluateEpochInternal",
              "evaluateBatchInternal", "inferenceWithLoss"]
LEAF_RE = re.compile(r"(Forward|Backward|Kernel|forward|backward)")
LIBC_MARGIN = 8 * 1024


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--build-dir", type=Path, required=True)
    args = p.parse_args()

    frames = {}
    for su in args.build_dir.rglob("*.su"):
        for line in su.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            fn = parts[0].split(":")[-1]
            try:
                size = int(parts[1])
            except ValueError:
                continue
            frames[fn] = max(frames.get(fn, 0), size)

    if not frames:
        print("no .su files found — build with -DCMAKE_C_FLAGS=-fstack-usage "
              "(AppleClang may lack support: run inside the Linux container)",
              file=sys.stderr)
        return 1

    def chain_sum(label: str, chain: list[str]) -> int:
        subtotal = 0
        for fn in chain:
            size = frames.get(fn)
            if size is None:
                print(f"chain fn {fn}: NOT FOUND in .su data (inlined or renamed) — using 0",
                      file=sys.stderr)
                size = 0
            print(f"{fn:32s} {size:8d} B")
            subtotal += size
        print(f"{'subtotal ' + label:32s} {subtotal:8d} B")
        return subtotal

    shared = chain_sum("SHARED", SHARED)
    train = chain_sum("TRAIN_CHAIN", TRAIN_CHAIN)
    eval_ = chain_sum("EVAL_CHAIN", EVAL_CHAIN)
    winner = "TRAIN_CHAIN" if train >= eval_ else "EVAL_CHAIN"
    print(f"{'deeper chain: ' + winner:32s} {max(train, eval_):8d} B")

    leaf_fn, leaf = max(((f, s) for f, s in frames.items() if LEAF_RE.search(f)),
                        key=lambda kv: kv[1], default=("<none>", 0))
    print(f"{'worst layer leaf: ' + leaf_fn:32s} {leaf:8d} B")
    print(f"{'libc margin':32s} {LIBC_MARGIN:8d} B")
    total = shared + max(train, eval_) + leaf + LIBC_MARGIN
    print(f"{'STATIC BOUND':32s} {total:8d} B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
