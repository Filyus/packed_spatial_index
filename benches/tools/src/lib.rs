//! Shared helpers for the local performance tools.
//!
//! Each tool emits one JSON object per line (JSONL) on stdout via [`emit`], so a
//! run can be parsed, diffed, or aggregated downstream (e.g. into the
//! performance.md tables) instead of scraping a printed table. Every row carries
//! a `tool` field naming its shape. Pipe through `jq` to read by eye.

use serde::Serialize;

/// Print one JSONL row.
pub fn emit<T: Serialize>(value: &T) {
    println!(
        "{}",
        serde_json::to_string(value).expect("perf-tool rows should serialize")
    );
}

/// Pin the current thread to the logical core named by `BENCH_PIN_CORE`, for
/// low-noise timing. No-op when unset; Windows-only self-pin (use `taskset` on
/// Linux). Mirrors the benchmark harness's `pin` helper.
pub fn pin_from_env() {
    let Ok(val) = std::env::var("BENCH_PIN_CORE") else {
        return;
    };
    let Ok(core) = val.trim().parse::<usize>() else {
        return;
    };
    if core >= usize::BITS as usize {
        return;
    }
    #[cfg(windows)]
    {
        unsafe extern "system" {
            fn GetCurrentThread() -> isize;
            fn SetThreadAffinityMask(thread: isize, mask: usize) -> usize;
        }
        // SAFETY: a pseudo-handle and a single in-range affinity bit; no memory
        // is touched.
        unsafe {
            SetThreadAffinityMask(GetCurrentThread(), 1usize << core);
        }
        eprintln!("perf: pinned to logical core {core}");
    }
    #[cfg(not(windows))]
    {
        eprintln!("perf: BENCH_PIN_CORE is Windows-only here; use `taskset -c {core}`");
    }
}
