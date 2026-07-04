use std::fs::File;
use std::path::{Path, PathBuf};

use packed_spatial_index_geo::{NullPolicy, ValidateRequest, open_geoparquet};

#[test]
#[ignore = "set PSI_GEO_CORPUS to an external fixture corpus to run this smoke"]
fn external_geo_corpus_validates_and_builds() {
    let root = std::env::var_os("PSI_GEO_CORPUS")
        .map(PathBuf::from)
        .expect("PSI_GEO_CORPUS must point to the external fixture corpus");
    let root = resolve_corpus_root(root);
    let mut files = Vec::new();
    collect_parquet_files(&root, &mut files);
    files.sort();
    assert!(
        !files.is_empty(),
        "no parquet files under {}",
        root.display()
    );

    for path in files {
        let mut dataset = open_geoparquet(File::open(&path).unwrap()).unwrap_or_else(|err| {
            panic!("open failed for {}: {err}", path.display());
        });
        let report = dataset
            .validate(ValidateRequest {
                exact: true,
                nulls: NullPolicy::Skip,
                ..ValidateRequest::default()
            })
            .unwrap_or_else(|err| panic!("validate failed for {}: {err}", path.display()));
        assert!(
            report.ok,
            "{} validation issues: {:?}",
            path.display(),
            report.issues
        );

        let mut dataset = open_geoparquet(File::open(&path).unwrap()).unwrap_or_else(|err| {
            panic!("reopen failed for {}: {err}", path.display());
        });
        dataset
            .build(packed_spatial_index_geo::BuildRequest {
                nulls: NullPolicy::Skip,
                ..Default::default()
            })
            .unwrap_or_else(|err| panic!("build failed for {}: {err}", path.display()));
    }
}

fn resolve_corpus_root(path: PathBuf) -> PathBuf {
    if path.is_absolute() || path.exists() {
        return path;
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_relative = manifest_dir
        .parent()
        .map(|parent| parent.join(&path))
        .unwrap_or_else(|| path.clone());
    if repo_relative.exists() {
        repo_relative
    } else {
        path
    }
}

fn collect_parquet_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap_or_else(|err| {
        panic!("cannot read {}: {err}", dir.display());
    }) {
        let path = entry.unwrap().path();
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name == "_smoke") {
                continue;
            }
            collect_parquet_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "parquet") {
            out.push(path);
        }
    }
}
