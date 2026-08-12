//! The `unsafe` inventory in SAFETY.md, held to the source by a count per file.
//!
//! SAFETY.md says `unsafe` "is confined to narrow, audited paths" and then lists
//! what those paths are. Nothing enforces that sentence: the code stays sound,
//! every block keeps its `// SAFETY:` comment, Clippy's `undocumented_unsafe_blocks`
//! keeps passing, and the only thing that breaks is the claim an auditor reads
//! instead of grepping. It had already broken — the list named AVX-512 while the
//! AVX2 paths, the SSE prefetch hint, the left-pack writes into reserved slack
//! and the payload record casts were all missing from it.
//!
//! So the counts are pinned here. They are deliberately counts and not a list of
//! names: a name list would be a second copy of the table in SAFETY.md, free to
//! drift the same way. A number cannot describe what it counts, so the only way
//! to move one is to open the file and look at what was added.
//!
//! When this fails, do not just change the number. Read the new block, decide
//! which SAFETY.md category it belongs to, and add the category if it is new.

use std::fs;
use std::path::{Path, PathBuf};

/// Files carrying `unsafe`, and how many sites each has. Everything else in
/// `src/` must have none.
const EXPECTED: &[(&str, usize)] = &[
    ("index2d.rs", 2),
    ("index2d_f32.rs", 12),
    ("index2d_soa/raycast.rs", 16),
    ("index2d_soa.rs", 16),
    ("index3d.rs", 2),
    ("index3d_f32.rs", 12),
    ("index3d_soa/raycast.rs", 16),
    ("index3d_soa.rs", 16),
    ("leftpack.rs", 2),
    ("persistence/mod.rs", 2),
    ("persistence/writer.rs", 5),
    ("traversal.rs", 2),
    ("triangle.rs", 2),
];

/// Occurrences of `unsafe` that introduce code — a block, a function, an `impl`,
/// a `trait` or an `extern` — outside the file's `mod tests`.
///
/// Line comments are stripped first, so a `// SAFETY:` note that says "unsafe"
/// does not count. Test modules are excluded because their `unsafe` is
/// scaffolding, and the audit is about what ships.
fn unsafe_sites(source: &str) -> usize {
    let head = match source.find("\n#[cfg(test)]\nmod tests {") {
        Some(at) => &source[..at],
        None => source,
    };

    let mut count = 0;
    for line in head.lines() {
        let code = line.split("//").next().unwrap_or("");
        let mut rest = code;
        while let Some(at) = rest.find("unsafe") {
            let preceded_by_word = rest[..at]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric() || c == '_');
            let after = rest[at + "unsafe".len()..].trim_start();
            let introduces_code = after.starts_with('{')
                || after.starts_with("fn ")
                || after.starts_with("impl ")
                || after.starts_with("trait ")
                || after.starts_with("extern ");
            if !preceded_by_word && introduces_code {
                count += 1;
            }
            rest = &rest[at + "unsafe".len()..];
        }
    }
    count
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("src/ is carried by the repository") {
        let path = entry.expect("readable directory entry").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn unsafe_inventory_matches_safety_md() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&src, &mut files);
    files.sort();

    let mut actual: Vec<(String, usize)> = Vec::new();
    for path in &files {
        let text = fs::read_to_string(path).expect("source file is valid UTF-8");
        let count = unsafe_sites(&text);
        if count > 0 {
            let rel = path
                .strip_prefix(&src)
                .expect("walked from src/")
                .to_string_lossy()
                .replace('\\', "/");
            actual.push((rel, count));
        }
    }

    let expected: Vec<(String, usize)> = EXPECTED
        .iter()
        .map(|&(name, n)| (name.to_string(), n))
        .collect();

    assert_eq!(
        actual, expected,
        "the `unsafe` inventory moved; see this file's header before editing the table"
    );
}

#[test]
fn counting_rule_ignores_comments_and_test_modules() {
    // Pins the rule itself: without this, a counter that silently matched nothing
    // would make the audit above pass forever.
    assert_eq!(unsafe_sites("// an unsafe { block } in a comment\n"), 0);
    assert_eq!(unsafe_sites("let not_unsafe = 1;\nfn unsafely() {}\n"), 0);
    assert_eq!(unsafe_sites("unsafe { x() }\nunsafe fn y() {}\n"), 2);
    assert_eq!(unsafe_sites("unsafe impl Send for T {}\n"), 1);
    assert_eq!(
        unsafe_sites("unsafe { a() }\n#[cfg(test)]\nmod tests {\n unsafe { b() }\n}\n"),
        1
    );
}
