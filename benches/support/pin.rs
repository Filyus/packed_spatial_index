//! Shared bench helper: optionally pin the measuring thread to one logical core
//! for low-noise numbers. Set `BENCH_PIN_CORE=<n>` before `cargo bench` (a fast
//! performance core; on a hybrid CPU avoid the efficiency cores — check your core
//! topology for which logical IDs are performance cores). No-op when the variable
//! is unset. Windows self-pins; elsewhere use the OS tool
//! (`taskset -c <n> cargo bench …` on Linux).
//!
//! Included by each bench via `#[path = "support/pin.rs"] mod pin;` and called
//! once at the top of `main`. It lives in a subdirectory so Cargo does not treat
//! it as its own benchmark target.

/// Pin to the core named by `BENCH_PIN_CORE`, if that variable is set.
pub fn pin_from_env() {
    let Ok(val) = std::env::var("BENCH_PIN_CORE") else {
        return;
    };
    match val.trim().parse::<usize>() {
        Ok(core) => pin_to_core(core),
        Err(_) => eprintln!("bench: BENCH_PIN_CORE=\"{val}\" is not a core number; not pinning"),
    }
}

/// Pin the current thread to a single logical core.
pub fn pin_to_core(core: usize) {
    if core >= usize::BITS as usize {
        eprintln!("bench: core {core} is out of range for an affinity mask; not pinning");
        return;
    }
    #[cfg(windows)]
    {
        unsafe extern "system" {
            fn GetCurrentThread() -> isize;
            fn SetThreadAffinityMask(thread: isize, mask: usize) -> usize;
        }
        // SAFETY: GetCurrentThread returns a pseudo-handle valid for the call;
        // the mask is a single in-range logical-core bit. No memory is accessed.
        let prev = unsafe { SetThreadAffinityMask(GetCurrentThread(), 1usize << core) };
        if prev == 0 {
            eprintln!("bench: failed to pin to logical core {core}");
        } else {
            eprintln!("bench: pinned measuring thread to logical core {core}");
        }
    }
    #[cfg(not(windows))]
    {
        eprintln!(
            "bench: BENCH_PIN_CORE is Windows-only here; on Linux run e.g. `taskset -c {core} cargo bench …`"
        );
    }
}
