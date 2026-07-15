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
