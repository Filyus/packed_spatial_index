#![cfg(feature = "geojson")]
//! The prefix section, end to end: written when the bodies are fat, skipped
//! when they are not, and cheap to scan either way.

use std::cell::Cell;
use std::io;
use std::rc::Rc;

use packed_spatial_index_geo::{
    Box2D, ConvertRequest, FEATURE_REF_RECORD_LEN, GeoArtifactIndex, PayloadPlan,
    PrefixIndexPolicy, PropertyProjection, RangeReader, open_geo_index, open_geojson_slice,
};

/// `n` points, each carrying `filler_len` bytes of properties, so the payload
/// body size is a test knob.
fn geojson(n: usize, filler_len: usize) -> Vec<u8> {
    let filler = "x".repeat(filler_len);
    let features: Vec<String> = (0..n)
        .map(|i| {
            let x = (i % 100) as f64 * 0.1;
            let y = (i / 100) as f64 * 0.1;
            format!(
                r#"{{"type":"Feature","id":"f{i}","geometry":{{"type":"Point","coordinates":[{x},{y}]}},"properties":{{"note":"{filler}"}}}}"#
            )
        })
        .collect();
    format!(
        r#"{{"type":"FeatureCollection","features":[{}]}}"#,
        features.join(",")
    )
    .into_bytes()
}

fn convert(source: &[u8], payload: PayloadPlan, prefix_index: PrefixIndexPolicy) -> Vec<u8> {
    let mut dataset = open_geojson_slice(source).unwrap();
    dataset
        .convert(ConvertRequest {
            payload,
            prefix_index,
            ..ConvertRequest::default()
        })
        .unwrap()
}

struct Counting {
    bytes: Vec<u8>,
    reads: Rc<Cell<usize>>,
}

impl RangeReader for Counting {
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.reads.set(self.reads.get() + 1);
        let start = offset as usize;
        let src = self
            .bytes
            .get(start..start + buf.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "out of range"))?;
        buf.copy_from_slice(src);
        Ok(())
    }
    fn len(&self) -> Option<u64> {
        Some(self.bytes.len() as u64)
    }
}

/// Reads issued by a header search over the whole extent.
fn header_search_reads(bytes: Vec<u8>) -> (usize, usize) {
    let reads = Rc::new(Cell::new(0));
    let reader = Counting {
        bytes,
        reads: Rc::clone(&reads),
    };
    let GeoArtifactIndex::D2(index) = open_geo_index(reader).unwrap() else {
        panic!("expected a 2D artifact");
    };
    let before = reads.get();
    let headers = index
        .search_match_headers(Box2D::new(-1.0, -1.0, 100.0, 100.0))
        .unwrap();
    (reads.get() - before, headers.len())
}

const N: usize = 500;

#[test]
fn fat_bodies_get_a_section_and_scan_in_runs() {
    let source = geojson(N, 400);
    let payload = PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    };

    let auto = convert(&source, payload.clone(), PrefixIndexPolicy::Auto);
    let never = convert(&source, payload, PrefixIndexPolicy::Never);

    // The section costs 24 bytes per entry, plus its 12-byte descriptor, a
    // 24-byte directory entry and alignment padding — and nothing else.
    let overhead = auto.len() - never.len();
    let body = N * FEATURE_REF_RECORD_LEN;
    assert!(
        (body..body + 64).contains(&overhead),
        "unexpected overhead {overhead} for a {body}-byte section"
    );

    let (with_reads, with_hits) = header_search_reads(auto);
    let (without_reads, without_hits) = header_search_reads(never);
    assert_eq!(with_hits, N);
    assert_eq!(without_hits, N);
    assert!(
        without_reads >= N,
        "the strided scan should read once per match, got {without_reads}"
    );
    assert!(
        with_reads * 10 < without_reads,
        "the section should collapse reads: {with_reads} vs {without_reads}"
    );
}

#[test]
fn lean_bodies_are_left_alone() {
    // A `row-wkb` point is 45 bytes with its feature ref — under the threshold,
    // so its prefixes already coalesce and the section would be dead weight.
    let source = geojson(N, 0);
    let auto = convert(&source, PayloadPlan::RowWkb, PrefixIndexPolicy::Auto);
    let never = convert(&source, PayloadPlan::RowWkb, PrefixIndexPolicy::Never);
    assert_eq!(auto.len(), never.len());

    let (auto_reads, hits) = header_search_reads(auto);
    assert_eq!(hits, N);
    assert!(
        auto_reads * 10 < N,
        "lean bodies should already scan in runs, got {auto_reads}"
    );
}

/// `N` two-point line strings: a 65-byte payload, which is the smallest shape
/// whose prefixes no longer coalesce. There is no middle ground on either side
/// of that cliff, so the section has to start right there.
fn line_strings(n: usize) -> Vec<u8> {
    let features: Vec<String> = (0..n)
        .map(|i| {
            let x = (i % 100) as f64 * 0.1;
            let y = (i / 100) as f64 * 0.1;
            format!(
                r#"{{"type":"Feature","geometry":{{"type":"LineString","coordinates":[[{x},{y}],[{},{y}]]}},"properties":{{}}}}"#,
                x + 0.001
            )
        })
        .collect();
    format!(
        r#"{{"type":"FeatureCollection","features":[{}]}}"#,
        features.join(",")
    )
    .into_bytes()
}

#[test]
fn a_payload_just_past_the_coalescing_window_gets_a_section() {
    let source = line_strings(N);
    let auto = convert(&source, PayloadPlan::RowWkb, PrefixIndexPolicy::Auto);
    let never = convert(&source, PayloadPlan::RowWkb, PrefixIndexPolicy::Never);

    // 24 bytes per entry and no more; this is the whole price.
    let overhead = auto.len() - never.len();
    let section = N * FEATURE_REF_RECORD_LEN;
    assert!(
        (section..section + 64).contains(&overhead),
        "unexpected overhead {overhead} for a {section}-byte section"
    );

    let (auto_reads, hits) = header_search_reads(auto);
    let (never_reads, _) = header_search_reads(never);
    assert_eq!(hits, N);
    assert!(
        never_reads >= N,
        "a 65-byte payload should scan one read per match, got {never_reads}"
    );
    assert!(
        auto_reads * 10 < never_reads,
        "the section should collapse reads: {auto_reads} vs {never_reads}"
    );
}

#[test]
fn a_row_ref_artifact_never_duplicates_its_payload() {
    let source = geojson(N, 0);
    let auto = convert(&source, PayloadPlan::RowRef, PrefixIndexPolicy::Auto);
    let forced = convert(&source, PayloadPlan::RowRef, PrefixIndexPolicy::Always);
    let never = convert(&source, PayloadPlan::RowRef, PrefixIndexPolicy::Never);
    assert_eq!(auto.len(), never.len());
    assert_eq!(forced.len(), never.len());
}

#[test]
fn the_cli_flag_reaches_the_artifact() {
    // Registering an option in only some of the CLI's five places is a real
    // failure mode here -- a missing entry in `option_takes_value` turns the
    // value into a stray positional -- so drive the binary rather than the API.
    let dir = std::env::temp_dir().join("psi-prefix-index-cli");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("in.geojson");
    std::fs::write(&source, geojson(N, 400)).unwrap();

    let build = |mode: &str, out: &std::path::Path| {
        let status = std::process::Command::new(env!("CARGO_BIN_EXE_gp2psindex"))
            .args(["build", source.to_str().unwrap(), out.to_str().unwrap()])
            .args(["--format", "geojson"])
            .args(["--payload", "feature-json"])
            .args(["--properties", "all"])
            .args(["--prefix-index", mode])
            .output()
            .unwrap();
        assert!(
            status.status.success(),
            "stderr: {}",
            String::from_utf8_lossy(&status.stderr)
        );
        std::fs::metadata(out).unwrap().len()
    };

    let on = build("on", &dir.join("on.psindex"));
    let off = build("off", &dir.join("off.psindex"));
    let auto = build("auto", &dir.join("auto.psindex"));
    assert!(on > off, "`on` should add a section: {on} vs {off}");
    assert_eq!(auto, on, "fat bodies should pick the section by themselves");

    let bad = std::process::Command::new(env!("CARGO_BIN_EXE_gp2psindex"))
        .args(["build", source.to_str().unwrap(), "unused.psindex"])
        .args(["--format", "geojson"])
        .args(["--prefix-index", "yes-please"])
        .output()
        .unwrap();
    assert!(!bad.status.success());
    assert!(
        String::from_utf8_lossy(&bad.stderr).contains("--prefix-index"),
        "stderr: {}",
        String::from_utf8_lossy(&bad.stderr)
    );
}

#[test]
fn the_geo_manifest_survives_the_extra_chunk() {
    // `append_geo_manifest` rewrites the container after the core writes it,
    // moving every chunk right to make room. A fourth optional chunk is exactly
    // what that relocation could get wrong.
    let source = geojson(N, 400);
    let bytes = convert(
        &source,
        PayloadPlan::FeatureJson {
            properties: PropertyProjection::AllNonGeometry,
        },
        PrefixIndexPolicy::Always,
    );
    let GeoArtifactIndex::D2(index) =
        open_geo_index(packed_spatial_index_geo::SliceReader::new(bytes.clone())).unwrap()
    else {
        panic!("expected a 2D artifact");
    };
    assert_eq!(index.manifest().feature_count, N);

    // Payload bodies still decode, so the prefixes really were copied rather
    // than moved out of the blobs.
    let headers = index
        .search_match_headers(Box2D::new(-1.0, -1.0, 100.0, 100.0))
        .unwrap();
    let matches = index.fetch_matches(&headers[..3]).unwrap();
    assert_eq!(matches.len(), 3);
    for m in &matches {
        assert!(m.feature.feature_id.is_some());
    }
}
