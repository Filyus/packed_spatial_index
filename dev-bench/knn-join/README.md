# kNN-join gate harness

The A/B behind the measurement in the commit that added this directory. Both
arms run in one binary — the dual-tree `knn_join` and the per-item
`neighbors_of_box` loop it was meant to replace — with the arm order alternated
per round and `knn_join` listed a second time as a control that should not move.

```
cargo build --release --manifest-path dev-bench/knn-join/Cargo.toml
taskset -c 2 dev-bench/knn-join/target/release/knnjoin-bench
```

It asserts the two arms agree on every row's distances before it times
anything, so it is also a correctness check.

Kept because the number it produced is a *negative* result: without the
harness, "the dual tree loses at k = 50" rests on nothing but this commit
message, and the next person to reach for the idea has to rebuild the
apparatus before they can disagree with it.
