#![cfg(feature = "geojson")]

use packed_spatial_index_geo::{
    AntimeridianPolicy, Box2D, ConvertRequest, EnvelopePolicy, FEATURE_REF_RECORD_LEN,
    GeoArtifactIndex, GeoError, GeoMatchHeader, GeoPayload, GeoQuery2D, PayloadPlan,
    PropertyProjection, SliceReader, open_geo_index, open_geojson_slice,
};

fn sample_geojson() -> &'static [u8] {
    br#"{
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "crossing",
                "geometry": {"type": "LineString", "coordinates": [[170.0, 0.0], [-170.0, 1.0]]},
                "properties": {"name": "crossing"}
            },
            {
                "type": "Feature",
                "id": "west",
                "geometry": {"type": "Point", "coordinates": [-5.0, 1.0]},
                "properties": {"name": "west"}
            },
            {
                "type": "Feature",
                "id": "east",
                "geometry": {"type": "Point", "coordinates": [25.0, 3.0]},
                "properties": {"name": "east"}
            }
        ]
    }"#
}

fn artifact_bytes(payload: PayloadPlan) -> Vec<u8> {
    let mut source = open_geojson_slice(sample_geojson()).unwrap();
    source
        .convert(ConvertRequest {
            envelope: EnvelopePolicy::Geographic {
                antimeridian: AntimeridianPolicy::Split,
            },
            payload,
            ..ConvertRequest::default()
        })
        .unwrap()
}

fn artifact(payload: PayloadPlan) -> GeoArtifactIndex<SliceReader<Vec<u8>>> {
    open_geo_index(SliceReader::new(artifact_bytes(payload))).unwrap()
}

fn world() -> Box2D {
    Box2D::new(-180.0, -10.0, 180.0, 10.0)
}

#[test]
fn headers_agree_with_matches_for_row_wkb() {
    let GeoArtifactIndex::D2(index) = artifact(PayloadPlan::RowWkb) else {
        panic!("expected 2D artifact");
    };

    let mut matches = index.search_matches(world()).unwrap();
    let mut headers = index.search_match_headers(world()).unwrap();
    assert_eq!(headers.len(), matches.len());

    matches.sort_by_key(|m| m.entry_id);
    headers.sort_by_key(|h| h.entry_id);
    for (m, h) in matches.iter().zip(&headers) {
        assert_eq!(m.entry_id, h.entry_id);
        assert_eq!(m.feature, h.feature);
        let GeoPayload::RowWkb(wkb) = &m.payload else {
            panic!("expected RowWkb payload");
        };
        assert_eq!(h.payload_len, FEATURE_REF_RECORD_LEN + wkb.len());
    }

    // fetch_matches materializes equal matches, preserving header order.
    let fetched = index.fetch_matches(&headers).unwrap();
    assert_eq!(fetched, matches);

    // A page (subset, custom order) keeps its order.
    let page = vec![headers[2].clone(), headers[0].clone()];
    let fetched = index.fetch_matches(&page).unwrap();
    assert_eq!(fetched.len(), 2);
    assert_eq!(fetched[0].entry_id, page[0].entry_id);
    assert_eq!(fetched[1].entry_id, page[1].entry_id);
}

#[test]
fn headers_agree_with_matches_for_row_ref() {
    let GeoArtifactIndex::D2(index) = artifact(PayloadPlan::RowRef) else {
        panic!("expected 2D artifact");
    };
    let matches = index.search_matches(world()).unwrap();
    let headers = index.search_match_headers(world()).unwrap();
    assert_eq!(headers.len(), matches.len());
    for h in &headers {
        assert_eq!(h.payload_len, FEATURE_REF_RECORD_LEN);
    }
    let fetched = index.fetch_matches(&headers).unwrap();
    let by_entry = |m: &packed_spatial_index_geo::GeoMatch| m.entry_id;
    let mut a = fetched;
    let mut b = matches;
    a.sort_by_key(by_entry);
    b.sort_by_key(by_entry);
    assert_eq!(a, b);
}

#[test]
fn headers_agree_with_matches_for_feature_json() {
    let GeoArtifactIndex::D2(index) = artifact(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    }) else {
        panic!("expected 2D artifact");
    };

    let mut matches = index.search_matches(world()).unwrap();
    let mut headers = index.search_match_headers(world()).unwrap();
    assert_eq!(headers.len(), matches.len());

    matches.sort_by_key(|m| m.entry_id);
    headers.sort_by_key(|h| h.entry_id);
    for (m, h) in matches.iter().zip(&headers) {
        assert_eq!(m.entry_id, h.entry_id);
        let mut expected = m.feature.clone();
        expected.feature_id = None;
        assert_eq!(expected, h.feature);
        let GeoPayload::FeatureJson(feature) = &m.payload else {
            panic!("expected FeatureJson payload");
        };
        assert_eq!(feature["type"], "Feature");
        assert!(h.payload_len > FEATURE_REF_RECORD_LEN);
    }

    let fetched = index.fetch_matches(&headers).unwrap();
    assert_eq!(fetched, matches);
}

#[test]
fn feature_json_prefix_handles_row_number_starting_with_json_brace() {
    let features: Vec<_> = (0..=123)
        .map(|row_number| {
            serde_json::json!({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [row_number, 0]},
                "properties": {"row_number": row_number},
            })
        })
        .collect();
    let geojson = serde_json::to_vec(&serde_json::json!({
        "type": "FeatureCollection",
        "features": features,
    }))
    .unwrap();
    let mut source = open_geojson_slice(&geojson).unwrap();
    let bytes = source
        .convert(ConvertRequest {
            payload: PayloadPlan::FeatureJson {
                properties: PropertyProjection::AllNonGeometry,
            },
            ..ConvertRequest::default()
        })
        .unwrap();
    let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes)).unwrap() else {
        panic!("expected 2D artifact");
    };

    let bbox = Box2D::new(123.0, 0.0, 123.0, 0.0);
    let matches = index.search_matches(bbox).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].feature.row_number, 123);

    let headers = index.search_match_headers(bbox).unwrap();
    assert_eq!(headers.len(), 1);
    assert_eq!(headers[0].feature.row_number, 123);
    assert_eq!(index.fetch_matches(&headers).unwrap(), matches);
}

#[test]
fn feature_json_rejects_prefix_body_identity_mismatch() {
    let mut bytes = artifact_bytes(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    });
    let from = br#""row_number":1"#;
    let matches: Vec<_> = bytes
        .windows(from.len())
        .enumerate()
        .filter_map(|(offset, window)| (window == from).then_some(offset))
        .collect();
    assert_eq!(matches.len(), 1, "expected one row-one JSON feature ref");
    bytes[matches[0] + from.len() - 1] = b'9';

    let GeoArtifactIndex::D2(index) = open_geo_index(SliceReader::new(bytes)).unwrap() else {
        panic!("expected 2D artifact");
    };
    let err = index.search_matches(world()).unwrap_err();
    assert!(matches!(
        err,
        GeoError::PayloadDecode(message) if message.contains("prefix") && message.contains("feature_ref")
    ));

    let headers = index.search_match_headers(world()).unwrap();
    let err = index.fetch_matches(&headers).unwrap_err();
    assert!(matches!(
        err,
        GeoError::PayloadDecode(message) if message.contains("prefix") && message.contains("feature_ref")
    ));
}

#[test]
fn header_dedupe_matches_feature_level_semantics() {
    let GeoArtifactIndex::D2(index) = artifact(PayloadPlan::RowWkb) else {
        panic!("expected 2D artifact");
    };
    let mut headers = index.search_match_headers(world()).unwrap();
    assert_eq!(headers.len(), 4, "split line has two entries");
    GeoMatchHeader::dedupe_by_feature(&mut headers);
    assert_eq!(headers.len(), 3);
    assert!(headers.iter().all(|h| h.feature.part.is_none()));

    let feature_matches = index.search_feature_matches(world()).unwrap();
    assert_eq!(
        headers.iter().map(|h| h.entry_id).collect::<Vec<_>>(),
        feature_matches
            .iter()
            .map(|m| m.entry_id)
            .collect::<Vec<_>>(),
        "header dedupe picks the same representatives"
    );

    let fetched = index.fetch_matches(&headers).unwrap();
    assert!(fetched.iter().all(|m| m.feature.part.is_none()));
    assert_eq!(
        fetched.iter().map(|m| m.entry_id).collect::<Vec<_>>(),
        feature_matches
            .iter()
            .map(|m| m.entry_id)
            .collect::<Vec<_>>()
    );
}

#[test]
fn fetch_matches_rejects_stale_match_headers() {
    let GeoArtifactIndex::D2(index) = artifact(PayloadPlan::RowWkb) else {
        panic!("expected 2D artifact");
    };
    let header = index.search_match_headers(world()).unwrap().remove(0);

    let mut wrong_len = header.clone();
    wrong_len.payload_len += 1;
    let err = index.fetch_matches(&[wrong_len]).unwrap_err();
    assert!(matches!(
        err,
        GeoError::PayloadDecode(message) if message.contains("payload length changed")
    ));

    let mut wrong_identity = header;
    wrong_identity.feature.row_number += 1;
    let err = index.fetch_matches(&[wrong_identity]).unwrap_err();
    assert!(matches!(
        err,
        GeoError::PayloadDecode(message) if message.contains("match header")
    ));
}

#[test]
fn headers_support_polygon_queries() {
    let GeoArtifactIndex::D2(index) = artifact(PayloadPlan::RowWkb) else {
        panic!("expected 2D artifact");
    };
    use packed_spatial_index_geo::geo_types::{Coord, LineString, Polygon};
    let triangle = Polygon::new(
        LineString::new(vec![
            Coord { x: -10.0, y: -1.0 },
            Coord { x: 0.0, y: 5.0 },
            Coord { x: 10.0, y: -1.0 },
            Coord { x: -10.0, y: -1.0 },
        ]),
        vec![],
    );
    let query = GeoQuery2D::polygon(triangle);
    let matches = index.search_matches(query.clone()).unwrap();
    let headers = index.search_match_headers(query).unwrap();
    assert_eq!(
        headers.iter().map(|h| h.entry_id).collect::<Vec<_>>(),
        matches.iter().map(|m| m.entry_id).collect::<Vec<_>>()
    );
}

#[test]
fn headers_reject_unsupported_plans() {
    let GeoArtifactIndex::D2(index) = artifact(PayloadPlan::None) else {
        panic!("expected 2D artifact");
    };
    assert!(matches!(
        index.search_match_headers(world()),
        Err(GeoError::UnsupportedArtifact(_))
    ));
}

#[cfg(feature = "async")]
struct AsyncSlice(Vec<u8>);

#[cfg(feature = "async")]
impl packed_spatial_index_geo::AsyncRangeReader for AsyncSlice {
    async fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        let start = usize::try_from(offset).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "offset out of range")
        })?;
        let end = start.checked_add(buf.len()).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "range overflow")
        })?;
        let source = self.0.get(start..end).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "range outside buffer")
        })?;
        buf.copy_from_slice(source);
        Ok(())
    }

    fn len(&self) -> Option<u64> {
        Some(self.0.len() as u64)
    }
}

#[cfg(feature = "async")]
#[test]
fn async_payload_header_pages_match_full_search_order() {
    let bytes = artifact_bytes(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    });
    let GeoArtifactIndex::D2(index) = pollster::block_on(
        packed_spatial_index_geo::open_geo_index_async(AsyncSlice(bytes)),
    )
    .unwrap() else {
        panic!("expected 2D artifact");
    };

    for query in [
        GeoQuery2D::from(world()),
        GeoQuery2D::spherical_radius(179.0, 0.0, 2_000_000.0),
    ] {
        let mut all =
            pollster::block_on(index.search_payload_headers_async(query.clone())).unwrap();
        packed_spatial_index_geo::GeoPayloadHeader::sort_by_entry(&mut all);

        for (offset, limit) in [(0, 0), (0, 2), (1, 2), (3, 10), (10, 2)] {
            let page = pollster::block_on(index.search_payload_headers_page_async(
                query.clone(),
                offset,
                limit,
            ))
            .unwrap();
            let expected: Vec<_> = all.iter().skip(offset).take(limit).cloned().collect();
            assert_eq!(page.number_matched, all.len());
            assert_eq!(page.headers, expected);
        }
    }
}

#[cfg(feature = "async")]
#[test]
fn async_payload_fetch_rejects_stale_header_length() {
    let bytes = artifact_bytes(PayloadPlan::FeatureJson {
        properties: PropertyProjection::AllNonGeometry,
    });
    let GeoArtifactIndex::D2(index) = pollster::block_on(
        packed_spatial_index_geo::open_geo_index_async(AsyncSlice(bytes)),
    )
    .unwrap() else {
        panic!("expected 2D artifact");
    };
    let mut header = pollster::block_on(index.search_payload_headers_async(world()))
        .unwrap()
        .remove(0);
    header.payload_len += 1;

    let err = pollster::block_on(index.fetch_payload_header_matches_async(&[header])).unwrap_err();
    assert!(matches!(
        err,
        GeoError::PayloadDecode(message) if message.contains("payload length changed")
    ));
}

#[cfg(feature = "async")]
#[test]
fn async_match_header_pages_match_full_search_order() {
    let bytes = artifact_bytes(PayloadPlan::RowWkb);
    let GeoArtifactIndex::D2(index) = pollster::block_on(
        packed_spatial_index_geo::open_geo_index_async(AsyncSlice(bytes)),
    )
    .unwrap() else {
        panic!("expected 2D artifact");
    };

    for query in [
        GeoQuery2D::from(world()),
        GeoQuery2D::spherical_radius(179.0, 0.0, 2_000_000.0),
    ] {
        let mut all = pollster::block_on(index.search_match_headers_async(query.clone())).unwrap();
        GeoMatchHeader::sort_by_entry(&mut all);

        for (offset, limit) in [(0, 0), (0, 2), (1, 2), (3, 10), (10, 2)] {
            let page = pollster::block_on(index.search_match_headers_page_async(
                query.clone(),
                offset,
                limit,
            ))
            .unwrap();
            let expected: Vec<_> = all.iter().skip(offset).take(limit).cloned().collect();
            assert_eq!(page.number_matched, all.len());
            assert_eq!(page.headers, expected);

            let matches = pollster::block_on(index.fetch_matches_async(&page.headers)).unwrap();
            assert_eq!(matches.len(), expected.len());
            assert!(
                matches
                    .iter()
                    .zip(&expected)
                    .all(|(matched, header)| matched.entry_id == header.entry_id)
            );
        }
    }
}
