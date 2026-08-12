"""Break one invariant, run the suite, restore. Records who notices.

Not a general mutation tester: each case below is a *plausible refactor bug*
against a named guard, chosen by reading the code rather than by permuting
tokens. It answers the question a green suite never does — which of these guards
is actually held by a test, and which one could be deleted tomorrow in silence.

    python tests/mutate.py                 # every case
    python tests/mutate.py shortcut        # cases whose label contains "shortcut"
    python tests/mutate.py --list          # what is covered, without running

Each case restores its file before the next one runs, including on Ctrl-C. The
tree must be clean before starting; the script refuses otherwise, because a
crash mid-run would otherwise be indistinguishable from your own edits.

**Reading the results.** "NOT CAUGHT" is not automatically a hole. Cases marked
`expect_caught=False` below are performance dials or equivalent mutants — the
code stays correct, only slower or differently shaped — and they are listed so
that nobody re-derives that from scratch. Everything else must say "caught".

Last run: 12 cases, 10 caught, 2 not caught, both expected.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = "parallel,simd,f32-storage,bench-internals,stream"


class Case:
    def __init__(self, label, path, old, new, breaks, expect_caught=True):
        self.label = label
        self.path = path
        self.old = old
        self.new = new
        self.breaks = breaks
        self.expect_caught = expect_caught


CASES = [
    Case(
        "shortcut-drops-overlap-2d",
        "src/index2d.rs",
        "if root.overlaps(query) && query.contains(root) {",
        "if query.contains(root) {",
        "a crossed box in a loaded file answers differently per entry point",
    ),
    Case(
        "shortcut-drops-overlap-shared",
        "src/range.rs",
        "if T::bounds_overlap(root, query) && T::bounds_contain(query, root) {",
        "if T::bounds_contain(query, root) {",
        "the same, for every view and the shared traversal",
    ),
    Case(
        "builder-accepts-crossed-bounds",
        "src/builder2d.rs",
        "        self.check_item_bounds()?;\n",
        "",
        "a box with min > max reaches a tree the search paths cannot agree on",
    ),
    Case(
        "overlaps-excludes-touching",
        "src/geometry.rs",
        "        (self.min_x <= other.max_x)\n            & (self.max_x >= other.min_x)",
        "        (self.min_x < other.max_x)\n            & (self.max_x >= other.min_x)",
        "boxes that touch along an edge stop matching",
    ),
    Case(
        "contained-flag-from-overlap",
        "src/index2d.rs",
        "let encoded_level = if query.contains(*b) {",
        "let encoded_level = if query.overlaps(*b) {",
        "a subtree is emitted whole when the query only clips it",
    ),
    Case(
        "leaf-index-bound-unchecked",
        "src/persistence/mod.rs",
        "        if index >= p.num_items {",
        "        if index > p.num_items {",
        "a leaf index one past the end loads and is handed to the caller",
    ),
    Case(
        "node-size-lower-bound",
        "src/tree.rs",
        "node_size.clamp(MIN_NODE_SIZE, MAX_NODE_SIZE)",
        "node_size.min(MAX_NODE_SIZE)",
        "node_size 0 or 1 builds a tree that cannot converge",
    ),
    Case(
        "simd-tail-block-skipped",
        "src/index2d_soa.rs",
        "while pos + 4 <= end {",
        "while pos + 4 < end {",
        "the last full SIMD block of a node is never tested",
    ),
    Case(
        "hilbert-coord-nan-arm",
        "src/sort2d.rs",
        "    if value.is_nan() {\n        0\n    } else if value > u16::MAX as f64 {",
        "    if value > u16::MAX as f64 {",
        "a NaN centre casts to an arbitrary sort key instead of 0",
    ),
    Case(
        "stack-drain-off-by-one",
        "src/range.rs",
        "        if stack.len() > 1 {\n            level = stack.pop().unwrap();",
        "        if stack.len() > 0 {\n            level = stack.pop().unwrap();",
        "the traversal pops a node index as if it were a level",
    ),
    # --- expected NOT CAUGHT, and why ---
    Case(
        "radix-sort-always-on",
        "src/sort2d.rs",
        "if radix && items.len() >= RADIX_SORT_MIN_ITEMS {",
        "if radix || items.len() >= RADIX_SORT_MIN_ITEMS {",
        "small inputs take the radix sort instead of the comparison sort",
        expect_caught=False,
    ),
    Case(
        "prefetch-hint-removed",
        "src/traversal.rs",
        "pub(crate) fn prefetch_read<T>(ptr: *const T) {",
        "pub(crate) fn prefetch_read<T>(ptr: *const T) {\n    if true {\n        return;\n    }",
        "the cache hint stops being issued",
        expect_caught=False,
    ),
]


def run_suite():
    """The suite, quiet. Returns True when everything passed."""
    result = subprocess.run(
        ["cargo", "test", "--features", FEATURES],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stdout + result.stderr


def noticing_tests(output):
    """Which test names failed, so a case reports who caught it."""
    names = re.findall(r"^\s{4}(\S+)$", output, re.MULTILINE)
    seen = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--list" in sys.argv:
        for case in CASES:
            mark = " " if case.expect_caught else "*"
            print(f"{mark} {case.label:34s} {case.path:24s} {case.breaks}")
        print("\n* = expected NOT CAUGHT (performance dial or equivalent mutant)")
        return 0

    dirty = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        print("working tree is not clean; commit or stash first:\n" + dirty)
        return 2

    cases = [c for c in CASES if not args or any(a in c.label for a in args)]
    print(f"running {len(cases)} case(s) against `cargo test --features {FEATURES}`\n")

    surprises = []
    for case in cases:
        path = os.path.join(ROOT, case.path)
        with open(path, encoding="utf-8") as f:
            original = f.read()
        if case.old not in original:
            print(f"{case.label:34s} SKIPPED (pattern is gone; the code moved)")
            surprises.append(case.label)
            continue

        backup = tempfile.NamedTemporaryFile(delete=False, suffix=".rs").name
        shutil.copyfile(path, backup)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(original.replace(case.old, case.new, 1))
            passed, output = run_suite()
        finally:
            shutil.copyfile(backup, path)
            os.unlink(backup)

        caught = not passed
        verdict = "caught" if caught else "NOT CAUGHT"
        by = ", ".join(noticing_tests(output)[:3]) if caught else ""
        expected = "" if caught == case.expect_caught else "   <-- unexpected"
        print(f"{case.label:34s} {verdict:11s} {by}{expected}")
        if caught != case.expect_caught:
            surprises.append(case.label)

    print()
    if surprises:
        print("unexpected: " + ", ".join(surprises))
        return 1
    print("every case matched its expectation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
