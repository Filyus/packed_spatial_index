//! GeoJSON (RFC 7946) source: open a GeoJSON document and build/convert
//! PSINDEX artifacts from it through the shared scan core.
//!
//! The eager dataset parses a document into memory for repeatable read-back.
//! For one-shot conversion/build of large FeatureCollections, use the
//! streaming helpers in this module.

use std::collections::HashSet;
use std::fmt;
use std::io::{self, Read};

use serde::Deserialize;
use serde::de::{self, DeserializeSeed, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde_json::value::RawValue;

use crate::payload::{self, FeatureRef};
use crate::scan_core::{self, FeatureReadRequest, FeatureRecord, GeometryReadMode, ScanEntry};
use crate::wkb::{self, Coord, GeometryBounds, GeometryParts};
use crate::{
    BuildRequest, ConvertRequest, CoordinateDims, CrsInfo, DeclaredExtent, EdgeModel,
    EnvelopePolicy, GeoArtifact, GeoError, GeoIndex, GeoSource, GeometryEncoding, GeometryKind,
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
pub fn open_geojson<R: Read>(mut reader: R) -> Result<GeoJsonDataset, GeoError> {
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .map_err(|e| GeoError::GeoJson(e.to_string()))?;
    open_geojson_slice(&bytes)
}

/// Convert a GeoJSON `FeatureCollection` from a reader without retaining the
/// full document or all parsed features.
///
/// This one-shot path accepts `FeatureCollection` documents only. Use
/// [`open_geojson`] for repeatable in-memory access or for single Feature /
/// bare geometry inputs.
///
/// # Example
///
/// ```
/// use packed_spatial_index_geo::{ConvertRequest, convert_geojson_stream};
///
/// let doc = br#"{"type":"FeatureCollection","features":[
///     {"type":"Feature","geometry":{"type":"Point","coordinates":[1.0,2.0]},"properties":{}}
/// ]}"#;
///
/// let mut bytes = Vec::new();
/// let artifact = convert_geojson_stream(&doc[..], ConvertRequest::default(), &mut bytes)?;
/// assert_eq!(artifact.manifest.index_entry_count, 1);
/// assert!(!bytes.is_empty());
/// # Ok::<(), packed_spatial_index_geo::GeoError>(())
/// ```
pub fn convert_geojson_stream<R: Read>(
    reader: R,
    req: ConvertRequest,
    out: &mut Vec<u8>,
) -> Result<GeoArtifact, GeoError> {
    let selector = req.selector.clone();
    let (scan, fingerprint) = scan_geojson_stream(
        reader,
        ScanRequest {
            selector,
            dims: req.dims,
            nulls: req.nulls,
            envelope: req.envelope,
            payload: req.payload.clone(),
        },
    )?;
    GeoArtifact::from_scan(&scan, &req, &fingerprint, out)
}

/// Build an in-memory index from a GeoJSON `FeatureCollection` without
/// retaining the full source document.
///
/// This one-shot path accepts `FeatureCollection` documents only. Use
/// [`open_geojson`] for repeatable in-memory access or for single Feature /
/// bare geometry inputs.
///
/// # Example
///
/// ```
/// use packed_spatial_index_geo::{Box2D, GeoIndex, build_geojson_stream};
///
/// let doc = br#"{"type":"FeatureCollection","features":[
///     {"type":"Feature","geometry":{"type":"Point","coordinates":[1.0,2.0]},"properties":{}}
/// ]}"#;
///
/// let GeoIndex::D2(index) = build_geojson_stream(&doc[..], Default::default())? else {
///     panic!("expected a 2D index");
/// };
/// let refs = index.search_feature_refs(Box2D::new(0.0, 0.0, 3.0, 3.0))?;
/// assert_eq!(refs.len(), 1);
/// # Ok::<(), packed_spatial_index_geo::GeoError>(())
/// ```
pub fn build_geojson_stream<R: Read>(reader: R, req: BuildRequest) -> Result<GeoIndex, GeoError> {
    let (scan, _) = scan_geojson_stream(
        reader,
        ScanRequest {
            selector: req.selector,
            dims: req.dims,
            nulls: req.nulls,
            envelope: req.envelope,
            payload: PayloadPlan::None,
        },
    )?;
    GeoIndex::from_scan(&scan, &req.build)
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
    let document: RawDocumentType =
        serde_json::from_slice(bytes).map_err(|e| GeoError::GeoJson(e.to_string()))?;
    let fingerprint = format!("fnv64:{:016x}", payload::fnv(0xcbf2_9ce4_8422_2325, bytes));
    let (features, extent) = match document.doc_type.as_str() {
        "FeatureCollection" => {
            let document: RawFeatureCollectionDocument =
                serde_json::from_slice(bytes).map_err(|e| GeoError::GeoJson(e.to_string()))?;
            let features = document.features.ok_or_else(|| {
                GeoError::GeoJson("FeatureCollection has no `features` array".to_string())
            })?;
            let features = features
                .into_iter()
                .enumerate()
                .map(|(row, feature)| parse_feature(feature, row))
                .collect::<Result<Vec<_>, _>>()?;
            (
                features,
                document.bbox.as_ref().and_then(declared_extent_from_bbox),
            )
        }
        "Feature" => {
            let feature: RawFeature =
                serde_json::from_slice(bytes).map_err(|e| GeoError::GeoJson(e.to_string()))?;
            (vec![parse_feature(feature, 0)?], None)
        }
        "Point" | "MultiPoint" | "LineString" | "MultiLineString" | "Polygon" | "MultiPolygon"
        | "GeometryCollection" => {
            let geometry = raw_value_from_slice(bytes)?;
            let geometry_type = raw_geometry_type(geometry.as_ref(), 0)?;
            (
                vec![GeoJsonFeature {
                    geometry: Some(geometry),
                    geometry_type: Some(geometry_type),
                    properties: None,
                    feature_id: None,
                }],
                None,
            )
        }
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

#[derive(Deserialize)]
struct RawDocumentType {
    #[serde(rename = "type")]
    doc_type: String,
}

#[derive(Deserialize)]
struct RawFeatureCollectionDocument {
    #[allow(dead_code)]
    #[serde(rename = "type")]
    doc_type: String,
    #[serde(default)]
    bbox: Option<serde_json::Value>,
    #[serde(default)]
    features: Option<Vec<RawFeature>>,
}

#[derive(Deserialize)]
struct RawFeature {
    #[serde(rename = "type")]
    feature_type: Option<String>,
    #[serde(default)]
    geometry: Option<Box<RawValue>>,
    #[serde(default)]
    properties: Option<Box<RawValue>>,
    #[serde(default)]
    id: Option<serde_json::Value>,
}

fn parse_feature(value: RawFeature, row: usize) -> Result<GeoJsonFeature, GeoError> {
    if value.feature_type.as_deref() != Some("Feature") {
        return Err(GeoError::GeoJson(format!(
            "features[{row}] is not a Feature object"
        )));
    }
    let (geometry, geometry_type) = match value.geometry {
        None => (None, None),
        Some(geometry) => {
            let geometry_type = raw_geometry_type(geometry.as_ref(), row)?;
            (Some(geometry), Some(geometry_type))
        }
    };
    let properties = match value.properties {
        None => None,
        Some(properties) => {
            validate_raw_properties(properties.as_ref(), row)?;
            Some(properties)
        }
    };
    let feature_id = feature_id(value.id.as_ref());
    Ok(GeoJsonFeature {
        geometry,
        geometry_type,
        properties,
        feature_id,
    })
}

fn raw_value_from_slice(bytes: &[u8]) -> Result<Box<RawValue>, GeoError> {
    let text = std::str::from_utf8(bytes).map_err(|e| GeoError::GeoJson(e.to_string()))?;
    RawValue::from_string(text.to_string()).map_err(|e| GeoError::GeoJson(e.to_string()))
}

fn raw_geometry_type(geometry: &RawValue, row: usize) -> Result<String, GeoError> {
    #[derive(Deserialize)]
    struct GeometryType {
        #[serde(rename = "type")]
        geometry_type: Option<String>,
    }

    let value: GeometryType = serde_json::from_str(geometry.get())
        .map_err(|e| geojson_feature_error(row, &e.to_string()))?;
    value
        .geometry_type
        .ok_or_else(|| geojson_feature_error(row, "geometry has no `type` member"))
}

fn validate_raw_properties(properties: &RawValue, row: usize) -> Result<(), GeoError> {
    if properties.get().trim_start().starts_with('{') {
        Ok(())
    } else {
        Err(GeoError::GeoJson(format!(
            "features[{row}] properties is not an object"
        )))
    }
}

fn parse_geometry_raw(geometry: &RawValue, row: usize) -> Result<serde_json::Value, GeoError> {
    serde_json::from_str(geometry.get()).map_err(|e| geojson_feature_error(row, &e.to_string()))
}

fn parse_properties_raw(
    properties: Option<&RawValue>,
    row: usize,
) -> Result<Option<serde_json::Map<String, serde_json::Value>>, GeoError> {
    let Some(properties) = properties else {
        return Ok(None);
    };
    validate_raw_properties(properties, row)?;
    let properties = serde_json::from_str(properties.get())
        .map_err(|e| geojson_feature_error(row, &e.to_string()))?;
    Ok(Some(properties))
}

fn project_properties_raw(
    properties: Option<&RawValue>,
    projection: &PropertyProjection,
    row: usize,
) -> Result<serde_json::Map<String, serde_json::Value>, GeoError> {
    if matches!(projection, PropertyProjection::None) {
        return Ok(serde_json::Map::new());
    }
    let properties = parse_properties_raw(properties, row)?;
    Ok(project_properties(properties.as_ref(), projection))
}

fn project_properties_raw_cached(
    properties: Option<&RawValue>,
    parsed: Option<&serde_json::Map<String, serde_json::Value>>,
    projection: &PropertyProjection,
    row: usize,
) -> Result<serde_json::Map<String, serde_json::Value>, GeoError> {
    if matches!(projection, PropertyProjection::None) {
        return Ok(serde_json::Map::new());
    }
    if let Some(properties) = parsed {
        return Ok(project_properties(Some(properties), projection));
    }
    project_properties_raw(properties, projection, row)
}

fn scan_geojson_stream<R: Read>(
    reader: R,
    req: ScanRequest,
) -> Result<(GeometryScan, String), GeoError> {
    check_geojson_selector(&req.selector)?;
    let mut reader = HashingReader::new(reader);
    let mut state = StreamScanState::new(&req);
    {
        let mut deserializer = serde_json::Deserializer::from_reader(&mut reader);
        FeatureCollectionSeed { state: &mut state }
            .deserialize(&mut deserializer)
            .map_err(|e| GeoError::GeoJson(e.to_string()))?;
        deserializer
            .end()
            .map_err(|e| GeoError::GeoJson(e.to_string()))?;
    }
    state.check_property_projection()?;
    let fingerprint = format!("fnv64:{:016x}", reader.hash);
    let profile = state.profile();
    let detected_dims = state.detected_dims;
    let entries = state.entries;
    Ok((
        scan_core::assemble_scan(entries, &req, profile, detected_dims)?,
        fingerprint,
    ))
}

struct HashingReader<R> {
    inner: R,
    hash: u64,
}

impl<R> HashingReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            hash: 0xcbf2_9ce4_8422_2325,
        }
    }
}

impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let len = self.inner.read(buf)?;
        self.hash = payload::fnv(self.hash, &buf[..len]);
        Ok(len)
    }
}

struct StreamScanState<'a> {
    req: &'a ScanRequest,
    entries: Vec<ScanEntry>,
    detected_dims: CoordinateDims,
    extent: Option<DeclaredExtent>,
    types: Vec<String>,
    seen_properties: HashSet<String>,
    row: usize,
}

impl<'a> StreamScanState<'a> {
    fn new(req: &'a ScanRequest) -> Self {
        Self {
            req,
            entries: Vec::new(),
            detected_dims: CoordinateDims::Unknown,
            extent: None,
            types: Vec::new(),
            seen_properties: HashSet::new(),
            row: 0,
        }
    }

    fn process_feature(&mut self, feature: RawFeature) -> Result<(), GeoError> {
        let row = self.row;
        self.row += 1;
        if feature.feature_type.as_deref() != Some("Feature") {
            return Err(GeoError::GeoJson(format!(
                "features[{row}] is not a Feature object"
            )));
        }
        let geometry = match feature.geometry {
            None => None,
            Some(geometry) => {
                let kind = raw_geometry_type(geometry.as_ref(), row)?;
                if !self.types.iter().any(|t| t == &kind) {
                    self.types.push(kind);
                }
                Some(geometry)
            }
        };
        if let Some(properties) = feature.properties.as_deref() {
            validate_raw_properties(properties, row)?;
        }
        let mut parsed_properties = None;
        if matches!(
            &self.req.payload,
            PayloadPlan::FeatureJson {
                properties: PropertyProjection::Include(_)
            }
        ) {
            parsed_properties = parse_properties_raw(feature.properties.as_deref(), row)?;
            if let Some(properties) = parsed_properties.as_ref() {
                self.seen_properties.extend(properties.keys().cloned());
            }
        }
        let geometry_value = geometry
            .as_deref()
            .map(|geometry| parse_geometry_raw(geometry, row))
            .transpose()?;
        let Some(bounds) = scan_geometry(
            geometry_value.as_ref(),
            row,
            matches!(self.req.envelope, EnvelopePolicy::Geographic { .. }),
        )?
        else {
            match self.req.nulls {
                NullPolicy::Skip => return Ok(()),
                NullPolicy::Error => return Err(GeoError::NullGeometry { row }),
            }
        };
        self.detected_dims = self.detected_dims.merge(bounds.dims);
        let mut feature_ref = FeatureRef::row_number(row as u64);
        feature_ref.feature_id = feature_id(feature.id.as_ref());
        let payload_bytes = match &self.req.payload {
            PayloadPlan::None => None,
            PayloadPlan::RowRef => Some(payload::encode_feature_ref(&feature_ref)),
            PayloadPlan::RowWkb => Some(payload::encode_feature_wkb(
                &feature_ref,
                &geometry_wkb(
                    geometry_value.as_ref().expect("bounds imply geometry"),
                    &bounds,
                    row,
                )?,
            )),
            PayloadPlan::FeatureJson {
                properties: projection,
            } => {
                let geometry = geometry.as_deref().expect("bounds imply geometry");
                let projected = project_properties_raw_cached(
                    feature.properties.as_deref(),
                    parsed_properties.as_ref(),
                    projection,
                    row,
                )?;
                Some(payload::feature_json_from_raw_parts(
                    &feature_ref,
                    geometry,
                    Some(serde_json::Value::Object(projected)),
                )?)
            }
        };
        self.entries.push(ScanEntry {
            bounds,
            feature: feature_ref,
            payload: payload_bytes,
        });
        Ok(())
    }

    fn check_property_projection(&self) -> Result<(), GeoError> {
        let PayloadPlan::FeatureJson {
            properties: PropertyProjection::Include(include),
        } = &self.req.payload
        else {
            return Ok(());
        };
        for name in include {
            if !self.seen_properties.contains(name) {
                return Err(GeoError::PropertyColumnNotFound(name.clone()));
            }
        }
        Ok(())
    }

    fn profile(&self) -> GeometryProfile {
        GeometryProfile {
            column: GEOMETRY_COLUMN.to_string(),
            source: GeometryMetadataSource::GeoJson,
            encoding: GeometryEncoding::GeoJson,
            crs: CrsInfo::ImpliedDefault {
                value: "OGC:CRS84".to_string(),
            },
            edges: EdgeModel::Planar,
            coordinate_dims: CoordinateDims::Unknown,
            geometry_types: GeometryTypeSet {
                types: self.types.clone(),
            },
            extent: self.extent.clone(),
            row_bounds: vec![RowBoundsSource::FeatureScan],
            num_rows: self.row as u64,
        }
    }
}

struct FeatureCollectionSeed<'a, 'req> {
    state: &'a mut StreamScanState<'req>,
}

impl<'de, 'a, 'req> DeserializeSeed<'de> for FeatureCollectionSeed<'a, 'req> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(FeatureCollectionVisitor { state: self.state })
    }
}

struct FeatureCollectionVisitor<'a, 'req> {
    state: &'a mut StreamScanState<'req>,
}

impl<'de, 'a, 'req> Visitor<'de> for FeatureCollectionVisitor<'a, 'req> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a GeoJSON FeatureCollection object")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let state = self.state;
        let mut doc_type: Option<String> = None;
        let mut saw_features = false;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "type" => doc_type = Some(map.next_value()?),
                "bbox" => {
                    let value: serde_json::Value = map.next_value()?;
                    state.extent = declared_extent_from_bbox(&value);
                }
                "features" => {
                    saw_features = true;
                    map.next_value_seed(FeaturesSeed { state: &mut *state })?;
                }
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        if doc_type.as_deref() != Some("FeatureCollection") {
            return Err(de::Error::custom(
                "document type is not `FeatureCollection`",
            ));
        }
        if !saw_features {
            return Err(de::Error::custom(
                "FeatureCollection has no `features` array",
            ));
        }
        Ok(())
    }
}

struct FeaturesSeed<'a, 'req> {
    state: &'a mut StreamScanState<'req>,
}

impl<'de, 'a, 'req> DeserializeSeed<'de> for FeaturesSeed<'a, 'req> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(FeaturesVisitor { state: self.state })
    }
}

struct FeaturesVisitor<'a, 'req> {
    state: &'a mut StreamScanState<'req>,
}

impl<'de, 'a, 'req> Visitor<'de> for FeaturesVisitor<'a, 'req> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a FeatureCollection features array")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while let Some(feature) = seq.next_element::<RawFeature>()? {
            self.state
                .process_feature(feature)
                .map_err(de::Error::custom)?;
        }
        Ok(())
    }
}

fn declared_extent_from_bbox(value: &serde_json::Value) -> Option<DeclaredExtent> {
    let bbox = value.as_array()?;
    let values: Vec<f64> = bbox.iter().filter_map(serde_json::Value::as_f64).collect();
    (values.len() == bbox.len() && (values.len() == 4 || values.len() == 6))
        .then_some(DeclaredExtent { values })
}

fn feature_id(value: Option<&serde_json::Value>) -> Option<String> {
    match value {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(id)) => Some(id.clone()),
        Some(other) => Some(other.to_string()),
    }
}

#[derive(Debug, Clone)]
struct GeoJsonFeature {
    geometry: Option<Box<RawValue>>,
    geometry_type: Option<String>,
    properties: Option<Box<RawValue>>,
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
            if let Some(kind) = feature.geometry_type.as_deref()
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
            let geometry = feature
                .geometry
                .as_deref()
                .map(|geometry| parse_geometry_raw(geometry, row))
                .transpose()?;
            let Some(bounds) = scan_geometry(geometry.as_ref(), row, collect_lons)? else {
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
                        geometry.as_ref().expect("bounds imply geometry"),
                        &bounds,
                        row,
                    )?,
                )),
                PayloadPlan::FeatureJson { properties } => {
                    let geometry = feature.geometry.as_deref().expect("bounds imply geometry");
                    let projected =
                        project_properties_raw(feature.properties.as_deref(), properties, row)?;
                    Some(payload::feature_json_from_raw_parts(
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
            .map(|feature| self.read_one(feature, req.geometry, req.geometry_json, &req.properties))
            .collect()
    }

    fn check_selector(&self, selector: &GeometrySelector) -> Result<(), GeoError> {
        check_geojson_selector(selector)
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
        let mut missing: HashSet<String> = include.iter().cloned().collect();
        for (row, feature) in self.features.iter().enumerate() {
            if missing.is_empty() {
                break;
            }
            if let Some(properties) = parse_properties_raw(feature.properties.as_deref(), row)? {
                missing.retain(|name| !properties.contains_key(name));
            }
        }
        if let Some(name) = missing.into_iter().next() {
            return Err(GeoError::PropertyColumnNotFound(name));
        }
        Ok(())
    }

    fn read_one(
        &self,
        mut feature_ref: FeatureRef,
        geometry: GeometryReadMode,
        geometry_json: bool,
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
            .ok_or(GeoError::FeatureRowOutOfBounds {
                row_number: feature_ref.row_number,
                num_rows: self.features.len() as u64,
            })?;
        if feature_ref.feature_id.is_none() {
            feature_ref.feature_id.clone_from(&source.feature_id);
        }
        let geometry_value = if (matches!(geometry, GeometryReadMode::Wkb) || geometry_json)
            && source.geometry.is_some()
        {
            Some(parse_geometry_raw(
                source.geometry.as_deref().expect("checked above"),
                row,
            )?)
        } else {
            None
        };
        let geometry_wkb = match (geometry, geometry_value.as_ref()) {
            (GeometryReadMode::Omit, _) | (_, None) => None,
            (GeometryReadMode::Wkb, Some(geometry)) => {
                Some(geometry_wkb_from_value(geometry, row)?)
            }
        };
        let properties = project_properties_raw(source.properties.as_deref(), properties, row)?;
        Ok(FeatureRecord {
            feature: feature_ref,
            geometry_wkb,
            geometry_json: geometry_json.then_some(geometry_value).flatten(),
            properties: serde_json::Value::Object(properties),
        })
    }
}

fn check_geojson_selector(selector: &GeometrySelector) -> Result<(), GeoError> {
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

fn scan_geometry(
    geometry: Option<&serde_json::Value>,
    row: usize,
    collect_lons: bool,
) -> Result<Option<GeometryBounds>, GeoError> {
    let Some(geometry) = geometry else {
        return Ok(None);
    };
    let mut bounds = GeometryBounds::new(collect_lons);
    visit_geometry_coords(geometry, row, &mut |coord| {
        bounds.add_coord(coord, collect_lons);
    })?;
    Ok(bounds.any.then_some(bounds))
}

fn geometry_wkb(
    geometry: &serde_json::Value,
    bounds: &GeometryBounds,
    row: usize,
) -> Result<Vec<u8>, GeoError> {
    let dims = if bounds.dims.has_z() {
        CoordinateDims::Xyz
    } else {
        CoordinateDims::Xy
    };
    let (kind, parts) = geometry_parts(geometry, row, dims)?;
    Ok(wkb::write_geometry(kind, dims, parts))
}

fn geometry_wkb_from_value(geometry: &serde_json::Value, row: usize) -> Result<Vec<u8>, GeoError> {
    let bounds =
        scan_geometry(Some(geometry), row, false)?.ok_or(GeoError::NullGeometry { row })?;
    geometry_wkb(geometry, &bounds, row)
}

fn visit_geometry_coords(
    geometry: &serde_json::Value,
    row: usize,
    f: &mut impl FnMut(&Coord),
) -> Result<(), GeoError> {
    match geometry_type(geometry, row)? {
        "Point" => {
            if let Some(coord) = coord_value(coordinates(geometry, row)?, row)? {
                f(&coord);
            }
        }
        "MultiPoint" | "LineString" => visit_coord_sequence(coordinates(geometry, row)?, row, f)?,
        "MultiLineString" | "Polygon" => visit_line_sequence(coordinates(geometry, row)?, row, f)?,
        "MultiPolygon" => {
            for polygon in array(coordinates(geometry, row)?, "coordinates", row)? {
                visit_line_sequence(polygon, row, f)?;
            }
        }
        "GeometryCollection" => {
            let geometries = geometry.get("geometries").ok_or_else(|| {
                geojson_feature_error(row, "GeometryCollection has no `geometries` array")
            })?;
            for child in array(geometries, "geometries", row)? {
                visit_geometry_coords(child, row, f)?;
            }
        }
        other => {
            return Err(geojson_feature_error(
                row,
                &format!("unsupported geometry type `{other}`"),
            ));
        }
    }
    Ok(())
}

fn visit_coord_sequence(
    value: &serde_json::Value,
    row: usize,
    f: &mut impl FnMut(&Coord),
) -> Result<(), GeoError> {
    for value in array(value, "coordinates", row)? {
        let Some(coord) = coord_value(value, row)? else {
            return Err(geojson_feature_error(
                row,
                "coordinate has fewer than two numbers",
            ));
        };
        f(&coord);
    }
    Ok(())
}

fn visit_line_sequence(
    value: &serde_json::Value,
    row: usize,
    f: &mut impl FnMut(&Coord),
) -> Result<(), GeoError> {
    for line in array(value, "coordinates", row)? {
        visit_coord_sequence(line, row, f)?;
    }
    Ok(())
}

fn geometry_parts(
    geometry: &serde_json::Value,
    row: usize,
    dims: CoordinateDims,
) -> Result<(GeometryKind, GeometryParts), GeoError> {
    match geometry_type(geometry, row)? {
        "Point" => Ok((
            GeometryKind::Point,
            GeometryParts::Point(
                coord_value(coordinates(geometry, row)?, row)?.unwrap_or_else(|| empty_point(dims)),
            ),
        )),
        "LineString" => Ok((
            GeometryKind::LineString,
            GeometryParts::LineString(coord_sequence(coordinates(geometry, row)?, row)?),
        )),
        "Polygon" => Ok((
            GeometryKind::Polygon,
            GeometryParts::Polygon(line_sequence(coordinates(geometry, row)?, row)?),
        )),
        "MultiPoint" => Ok((
            GeometryKind::MultiPoint,
            GeometryParts::LineString(coord_sequence(coordinates(geometry, row)?, row)?),
        )),
        "MultiLineString" => Ok((
            GeometryKind::MultiLineString,
            GeometryParts::Polygon(line_sequence(coordinates(geometry, row)?, row)?),
        )),
        "MultiPolygon" => {
            let mut polygons = Vec::new();
            for polygon in array(coordinates(geometry, row)?, "coordinates", row)? {
                polygons.push(line_sequence(polygon, row)?);
            }
            Ok((
                GeometryKind::MultiPolygon,
                GeometryParts::MultiPolygon(polygons),
            ))
        }
        "GeometryCollection" => {
            let geometries = geometry.get("geometries").ok_or_else(|| {
                geojson_feature_error(row, "GeometryCollection has no `geometries` array")
            })?;
            let mut children = Vec::new();
            for child in array(geometries, "geometries", row)? {
                children.push(geometry_parts(child, row, dims)?);
            }
            Ok((
                GeometryKind::Unknown,
                GeometryParts::GeometryCollection(children),
            ))
        }
        other => Err(geojson_feature_error(
            row,
            &format!("unsupported geometry type `{other}`"),
        )),
    }
}

fn coord_sequence(value: &serde_json::Value, row: usize) -> Result<Vec<Coord>, GeoError> {
    let values = array(value, "coordinates", row)?;
    let mut coords = Vec::with_capacity(values.len());
    for value in values {
        let Some(coord) = coord_value(value, row)? else {
            return Err(geojson_feature_error(
                row,
                "coordinate has fewer than two numbers",
            ));
        };
        coords.push(coord);
    }
    Ok(coords)
}

fn line_sequence(value: &serde_json::Value, row: usize) -> Result<Vec<Vec<Coord>>, GeoError> {
    let values = array(value, "coordinates", row)?;
    let mut lines = Vec::with_capacity(values.len());
    for value in values {
        lines.push(coord_sequence(value, row)?);
    }
    Ok(lines)
}

fn geometry_type(geometry: &serde_json::Value, row: usize) -> Result<&str, GeoError> {
    geometry
        .get("type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| geojson_feature_error(row, "geometry has no `type` member"))
}

fn coordinates(geometry: &serde_json::Value, row: usize) -> Result<&serde_json::Value, GeoError> {
    geometry
        .get("coordinates")
        .ok_or_else(|| geojson_feature_error(row, "geometry has no `coordinates` member"))
}

fn array<'a>(
    value: &'a serde_json::Value,
    context: &str,
    row: usize,
) -> Result<&'a Vec<serde_json::Value>, GeoError> {
    value
        .as_array()
        .ok_or_else(|| geojson_feature_error(row, &format!("{context} is not an array")))
}

fn coord_value(value: &serde_json::Value, row: usize) -> Result<Option<Coord>, GeoError> {
    let values = array(value, "coordinate", row)?;
    if values.is_empty() {
        return Ok(None);
    }
    if values.len() < 2 {
        return Err(geojson_feature_error(
            row,
            "coordinate has fewer than two numbers",
        ));
    }
    let x = finite_coord(values.first(), row, "x")?;
    let y = finite_coord(values.get(1), row, "y")?;
    let z = values
        .get(2)
        .map(|value| finite_coord(Some(value), row, "z"))
        .transpose()?;
    Ok(Some(Coord { x, y, z, m: None }))
}

fn finite_coord(
    value: Option<&serde_json::Value>,
    row: usize,
    axis: &str,
) -> Result<f64, GeoError> {
    let value = value
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| geojson_feature_error(row, &format!("coordinate {axis} is not a number")))?;
    if !value.is_finite() {
        return Err(geojson_feature_error(
            row,
            "geometry contains a non-finite coordinate",
        ));
    }
    Ok(value)
}

fn empty_point(dims: CoordinateDims) -> Coord {
    Coord {
        x: f64::NAN,
        y: f64::NAN,
        z: dims.has_z().then_some(f64::NAN),
        m: None,
    }
}

fn geojson_feature_error(row: usize, message: &str) -> GeoError {
    GeoError::GeoJson(format!("features[{row}]: {message}"))
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
