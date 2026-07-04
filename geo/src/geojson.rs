//! GeoJSON (RFC 7946) source: open a GeoJSON document and build/convert
//! PSINDEX artifacts from it through the shared scan core.
//!
//! The whole document is parsed into memory up front — GeoJSON has no row
//! groups or metadata footer to stream from, and the source documents this
//! crate targets (fixture sets, API responses, hand-maintained layers) are
//! small next to the Parquet datasets the `parquet` feature handles. A
//! streaming reader can come later without changing this module's API.

use geozero::GeozeroGeometry;
use geozero::geojson::GeoJson;

use crate::payload::{self, FeatureRef};
use crate::scan_core::{self, FeatureReadRequest, FeatureRecord, GeometryReadMode, ScanEntry};
use crate::wkb::{self, GeometryBounds};
use crate::{
    BuildRequest, ConvertRequest, CoordinateDims, CrsInfo, DeclaredExtent, EdgeModel,
    EnvelopePolicy, GeoArtifact, GeoError, GeoIndex, GeoSource, GeometryEncoding,
    GeometryMetadataSource, GeometryProfile, GeometryScan, GeometrySelector, GeometryTypeSet,
    NullPolicy, PayloadPlan, PropertyProjection, RowBoundsSource, ScanRequest,
};

/// Name under which the (single, implicit) GeoJSON geometry is selectable.
const GEOMETRY_COLUMN: &str = "geometry";

/// Open a GeoJSON document from a reader.
///
/// Accepts a `FeatureCollection`, a single `Feature`, or a bare geometry
/// object. The whole document is read and parsed eagerly; feature row numbers
/// are 0-based positions in source order.
///
/// # Example
///
/// ```no_run
/// use std::fs::File;
/// use packed_spatial_index_geo::open_geojson;
///
/// let mut dataset = open_geojson(File::open("cities.geojson")?)?;
/// println!("{} features", dataset.profile()?.num_rows);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn open_geojson<R: std::io::Read>(mut reader: R) -> Result<GeoJsonDataset, GeoError> {
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .map_err(|e| GeoError::GeoJson(e.to_string()))?;
    open_geojson_slice(&bytes)
}

/// Open a GeoJSON document from an in-memory byte slice.
///
/// See [`open_geojson`] for the accepted document shapes.
///
/// # Example
///
/// ```
/// use packed_spatial_index_geo::open_geojson_slice;
///
/// let doc = r#"{"type":"FeatureCollection","features":[
///     {"type":"Feature","geometry":{"type":"Point","coordinates":[1.0,2.0]},"properties":{"name":"a"}}
/// ]}"#;
/// let dataset = open_geojson_slice(doc.as_bytes())?;
/// assert_eq!(dataset.profile()?.num_rows, 1);
/// # Ok::<(), packed_spatial_index_geo::GeoError>(())
/// ```
pub fn open_geojson_slice(bytes: &[u8]) -> Result<GeoJsonDataset, GeoError> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| GeoError::GeoJson(e.to_string()))?;
    let fingerprint = format!("fnv64:{:016x}", payload::fnv(0xcbf2_9ce4_8422_2325, bytes));
    let doc_type = value
        .get("type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| GeoError::GeoJson("document has no `type` member".to_string()))?;
    let extent = declared_extent(&value);
    let features = match doc_type {
        "FeatureCollection" => {
            let features = value
                .get("features")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    GeoError::GeoJson("FeatureCollection has no `features` array".to_string())
                })?;
            features
                .iter()
                .enumerate()
                .map(|(row, feature)| parse_feature(feature, row))
                .collect::<Result<Vec<_>, _>>()?
        }
        "Feature" => vec![parse_feature(&value, 0)?],
        "Point" | "MultiPoint" | "LineString" | "MultiLineString" | "Polygon" | "MultiPolygon"
        | "GeometryCollection" => vec![GeoJsonFeature {
            geometry: Some(value.clone()),
            properties: None,
            feature_id: None,
        }],
        other => {
            return Err(GeoError::GeoJson(format!(
                "unsupported document type `{other}`"
            )));
        }
    };
    Ok(GeoJsonDataset {
        features,
        fingerprint,
        extent,
    })
}

fn declared_extent(value: &serde_json::Value) -> Option<DeclaredExtent> {
    let bbox = value.get("bbox")?.as_array()?;
    let values: Vec<f64> = bbox.iter().filter_map(serde_json::Value::as_f64).collect();
    (values.len() == bbox.len() && (values.len() == 4 || values.len() == 6))
        .then_some(DeclaredExtent { values })
}

fn parse_feature(value: &serde_json::Value, row: usize) -> Result<GeoJsonFeature, GeoError> {
    let feature_type = value.get("type").and_then(serde_json::Value::as_str);
    if feature_type != Some("Feature") {
        return Err(GeoError::GeoJson(format!(
            "features[{row}] is not a Feature object"
        )));
    }
    let geometry = match value.get("geometry") {
        None | Some(serde_json::Value::Null) => None,
        Some(geometry) => {
            if !geometry
                .get("type")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|_| true)
            {
                return Err(GeoError::GeoJson(format!(
                    "features[{row}] geometry has no `type` member"
                )));
            }
            Some(geometry.clone())
        }
    };
    let properties = match value.get("properties") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::Object(map)) => Some(map.clone()),
        Some(_) => {
            return Err(GeoError::GeoJson(format!(
                "features[{row}] properties is not an object"
            )));
        }
    };
    let feature_id = match value.get("id") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(id)) => Some(id.clone()),
        Some(other) => Some(other.to_string()),
    };
    Ok(GeoJsonFeature {
        geometry,
        properties,
        feature_id,
    })
}

#[derive(Debug, Clone)]
struct GeoJsonFeature {
    geometry: Option<serde_json::Value>,
    properties: Option<serde_json::Map<String, serde_json::Value>>,
    feature_id: Option<String>,
}

/// An opened GeoJSON document, ready to scan, build, or convert.
///
/// Unlike the Parquet `GeoDataset`, the parsed features
/// stay in memory, so scan/build/convert calls do not consume the dataset and
/// can be repeated.
#[derive(Debug, Clone)]
pub struct GeoJsonDataset {
    features: Vec<GeoJsonFeature>,
    fingerprint: String,
    extent: Option<DeclaredExtent>,
}

impl GeoJsonDataset {
    /// Profile of the (implicit) GeoJSON geometry column.
    ///
    /// GeoJSON documents always carry lon/lat WGS 84 coordinates (RFC 7946
    /// removed CRS negotiation), so the CRS is reported as the implied
    /// default `OGC:CRS84` and the edge model as planar. Coordinate
    /// dimensions are reported as unknown until a scan detects them.
    ///
    /// Returns `Result` for signature parity with the other sources (see
    /// [`GeoSource::profile`](crate::GeoSource::profile)); GeoJSON profiling
    /// never fails.
    pub fn profile(&self) -> Result<GeometryProfile, GeoError> {
        let mut types: Vec<String> = Vec::new();
        for feature in &self.features {
            if let Some(kind) = feature
                .geometry
                .as_ref()
                .and_then(|geometry| geometry.get("type"))
                .and_then(serde_json::Value::as_str)
                && !types.iter().any(|t| t == kind)
            {
                types.push(kind.to_string());
            }
        }
        Ok(GeometryProfile {
            column: GEOMETRY_COLUMN.to_string(),
            source: GeometryMetadataSource::GeoJson,
            encoding: GeometryEncoding::GeoJson,
            crs: CrsInfo::ImpliedDefault {
                value: "OGC:CRS84".to_string(),
            },
            edges: EdgeModel::Planar,
            coordinate_dims: CoordinateDims::Unknown,
            geometry_types: GeometryTypeSet { types },
            extent: self.extent.clone(),
            row_bounds: vec![RowBoundsSource::FeatureScan],
            num_rows: self.features.len() as u64,
        })
    }

    /// Stable fingerprint of the source document (FNV-64 over the raw bytes).
    pub fn source_fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// Scan feature envelopes, feature references, and optional payloads.
    ///
    /// Mirrors `GeoDataset::scan`: the same
    /// [`ScanRequest`] policies apply, including geographic antimeridian
    /// splitting — GeoJSON coordinates are always lon/lat, so
    /// [`EnvelopePolicy::Geographic`] is always legal.
    pub fn scan(&mut self, req: ScanRequest) -> Result<GeometryScan, GeoError> {
        self.check_selector(&req.selector)?;
        if let PayloadPlan::FeatureJson { properties } = &req.payload {
            self.check_property_projection(properties)?;
        }
        let collect_lons = matches!(req.envelope, EnvelopePolicy::Geographic { .. });
        let mut entries = scan_core::vec_with_capacity_hint(self.features.len());
        let mut detected_dims = CoordinateDims::Unknown;
        for (row, feature) in self.features.iter().enumerate() {
            let Some(bounds) = scan_geometry(feature.geometry.as_ref(), row, collect_lons)? else {
                match req.nulls {
                    NullPolicy::Skip => continue,
                    NullPolicy::Error => return Err(GeoError::NullGeometry { row }),
                }
            };
            detected_dims = detected_dims.merge(bounds.dims);
            let mut feature_ref = FeatureRef::row_number(row as u64);
            feature_ref.feature_id = feature.feature_id.clone();
            let payload_bytes = match &req.payload {
                PayloadPlan::None => None,
                PayloadPlan::RowRef => Some(payload::encode_feature_ref(&feature_ref)),
                PayloadPlan::RowWkb => Some(payload::encode_feature_wkb(
                    &feature_ref,
                    &geometry_wkb(
                        feature.geometry.as_ref().expect("bounds imply geometry"),
                        &bounds,
                        row,
                    )?,
                )),
                PayloadPlan::FeatureJson { properties } => {
                    let geometry = feature.geometry.clone().expect("bounds imply geometry");
                    let projected = project_properties(feature.properties.as_ref(), properties);
                    Some(payload::feature_json_from_parts(
                        &feature_ref,
                        geometry,
                        Some(serde_json::Value::Object(projected)),
                    )?)
                }
            };
            entries.push(ScanEntry {
                bounds,
                feature: feature_ref,
                payload: payload_bytes,
            });
        }
        scan_core::assemble_scan(entries, &req, self.profile()?, detected_dims)
    }

    /// Build an in-memory [`GeoIndex`] over the document's features.
    ///
    /// Mirrors `GeoDataset::build`.
    pub fn build(&mut self, req: BuildRequest) -> Result<GeoIndex, GeoError> {
        let scan = self.scan(ScanRequest {
            selector: req.selector,
            dims: req.dims,
            nulls: req.nulls,
            envelope: req.envelope,
            payload: PayloadPlan::None,
        })?;
        GeoIndex::from_scan(&scan, &req.build)
    }

    /// Convert the document into a streamable `PSINDEX` buffer.
    ///
    /// Mirrors `GeoDataset::convert_into`;
    /// the artifact manifest records `source_format: "geojson"`.
    pub fn convert_into(
        &mut self,
        req: ConvertRequest,
        out: &mut Vec<u8>,
    ) -> Result<GeoArtifact, GeoError> {
        let selector = req.selector.clone();
        let scan = self.scan(ScanRequest {
            selector,
            dims: req.dims,
            nulls: req.nulls,
            envelope: req.envelope,
            payload: req.payload.clone(),
        })?;
        let fingerprint = self.fingerprint.clone();
        GeoArtifact::from_scan(&scan, &req, &fingerprint, out)
    }

    /// Convert the document into a new `Vec<u8>`.
    ///
    /// Convenience wrapper around [`GeoJsonDataset::convert_into`].
    pub fn convert(&mut self, req: ConvertRequest) -> Result<Vec<u8>, GeoError> {
        let mut out = Vec::new();
        self.convert_into(req, &mut out)?;
        Ok(out)
    }

    /// Read source features back by [`FeatureRef`](crate::FeatureRef).
    ///
    /// GeoJSON documents are parsed eagerly, so this does not consume the
    /// dataset. Returned records keep source JSON property types intact.
    pub fn read_features(&self, req: FeatureReadRequest) -> Result<Vec<FeatureRecord>, GeoError> {
        self.check_selector(&req.selector)?;
        self.check_expected_fingerprint(req.expected_source_fingerprint.as_ref())?;
        self.check_property_projection(&req.properties)?;
        let output = scan_core::ordered_feature_refs(
            &req.features,
            Some(self.features.len() as u64),
            req.order,
            req.duplicates,
        )?;
        output
            .into_iter()
            .map(|feature| self.read_one(feature, req.geometry, &req.properties))
            .collect()
    }

    fn check_selector(&self, selector: &GeometrySelector) -> Result<(), GeoError> {
        match selector {
            GeometrySelector::Default | GeometrySelector::FirstUsable => Ok(()),
            GeometrySelector::Name(name) if name == GEOMETRY_COLUMN => Ok(()),
            GeometrySelector::Name(name) => Err(GeoError::GeometryColumnNotFound(name.clone())),
            GeometrySelector::GeoParquetPrimary | GeometrySelector::SingleNativeParquet => {
                Err(GeoError::Metadata(
                    "selector applies to Parquet sources; use Default or Name(\"geometry\") for GeoJSON".to_string(),
                ))
            }
        }
    }

    fn check_expected_fingerprint(&self, expected: Option<&String>) -> Result<(), GeoError> {
        if let Some(expected) = expected
            && expected != &self.fingerprint
        {
            return Err(GeoError::SourceFingerprintMismatch {
                expected: expected.clone(),
                actual: self.fingerprint.clone(),
            });
        }
        Ok(())
    }

    fn check_property_projection(&self, projection: &PropertyProjection) -> Result<(), GeoError> {
        let PropertyProjection::Include(include) = projection else {
            return Ok(());
        };
        for name in include {
            let known = self.features.iter().any(|feature| {
                feature
                    .properties
                    .as_ref()
                    .is_some_and(|properties| properties.contains_key(name))
            });
            if !known {
                return Err(GeoError::PropertyColumnNotFound(name.clone()));
            }
        }
        Ok(())
    }

    fn read_one(
        &self,
        mut feature_ref: FeatureRef,
        geometry: GeometryReadMode,
        properties: &PropertyProjection,
    ) -> Result<FeatureRecord, GeoError> {
        let row = usize::try_from(feature_ref.row_number).map_err(|_| {
            GeoError::FeatureRowOutOfBounds {
                row_number: feature_ref.row_number,
                num_rows: self.features.len() as u64,
            }
        })?;
        let source = self
            .features
            .get(row)
            .ok_or_else(|| GeoError::FeatureRowOutOfBounds {
                row_number: feature_ref.row_number,
                num_rows: self.features.len() as u64,
            })?;
        if feature_ref.feature_id.is_none() {
            feature_ref.feature_id.clone_from(&source.feature_id);
        }
        let geometry_wkb = match (geometry, source.geometry.as_ref()) {
            (GeometryReadMode::Omit, _) | (_, None) => None,
            (GeometryReadMode::Wkb, Some(geometry)) => {
                Some(geometry_wkb_from_value(geometry, row)?)
            }
        };
        Ok(FeatureRecord {
            feature: feature_ref,
            geometry_wkb,
            geometry_json: source.geometry.clone(),
            properties: serde_json::Value::Object(project_properties(
                source.properties.as_ref(),
                properties,
            )),
        })
    }
}

fn scan_geometry(
    geometry: Option<&serde_json::Value>,
    row: usize,
    collect_lons: bool,
) -> Result<Option<GeometryBounds>, GeoError> {
    let Some(geometry) = geometry else {
        return Ok(None);
    };
    let text = geometry.to_string();
    let geojson = GeoJson(&text);
    wkb::bounds_from_geozero(|processor| geojson.process_geom(processor), collect_lons)
        .map_err(|message| GeoError::GeoJson(format!("features[{row}]: {message}")))
}

fn geometry_wkb(
    geometry: &serde_json::Value,
    bounds: &GeometryBounds,
    row: usize,
) -> Result<Vec<u8>, GeoError> {
    use geozero::ToWkb;
    let dims = if bounds.dims.has_z() {
        geozero::CoordDimensions::xyz()
    } else {
        geozero::CoordDimensions::xy()
    };
    let text = geometry.to_string();
    GeoJson(&text)
        .to_wkb(dims)
        .map_err(|e| GeoError::GeoJson(format!("features[{row}]: {e}")))
}

fn geometry_wkb_from_value(geometry: &serde_json::Value, row: usize) -> Result<Vec<u8>, GeoError> {
    let bounds =
        scan_geometry(Some(geometry), row, false)?.ok_or(GeoError::NullGeometry { row })?;
    geometry_wkb(geometry, &bounds, row)
}

fn project_properties(
    properties: Option<&serde_json::Map<String, serde_json::Value>>,
    projection: &PropertyProjection,
) -> serde_json::Map<String, serde_json::Value> {
    let Some(properties) = properties else {
        return serde_json::Map::new();
    };
    match projection {
        PropertyProjection::None => serde_json::Map::new(),
        PropertyProjection::AllNonGeometry => properties.clone(),
        PropertyProjection::Include(include) => properties
            .iter()
            .filter(|(key, _)| include.iter().any(|name| name == *key))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
        PropertyProjection::Exclude(exclude) => properties
            .iter()
            .filter(|(key, _)| !exclude.iter().any(|name| name == *key))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    }
}

impl GeoSource for GeoJsonDataset {
    fn profile(&self) -> Result<GeometryProfile, GeoError> {
        GeoJsonDataset::profile(self)
    }

    fn source_fingerprint(&self) -> &str {
        GeoJsonDataset::source_fingerprint(self)
    }

    fn scan(&mut self, req: ScanRequest) -> Result<GeometryScan, GeoError> {
        GeoJsonDataset::scan(self, req)
    }

    fn build(&mut self, req: BuildRequest) -> Result<GeoIndex, GeoError> {
        GeoJsonDataset::build(self, req)
    }

    fn convert(&mut self, req: ConvertRequest) -> Result<Vec<u8>, GeoError> {
        GeoJsonDataset::convert(self, req)
    }

    fn convert_into(
        &mut self,
        req: ConvertRequest,
        out: &mut Vec<u8>,
    ) -> Result<GeoArtifact, GeoError> {
        GeoJsonDataset::convert_into(self, req, out)
    }
}
