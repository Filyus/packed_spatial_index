//! FlatGeobuf source: open an `.fgb` file and build/convert PSINDEX
//! artifacts from it through the shared scan core.
//!
//! The FlatGeobuf header supplies real metadata up front — geometry type,
//! Z/M dimensions, CRS, envelope, column schema, feature count — so
//! [`FgbDataset::profile`] is populated without touching feature data, like
//! Parquet discovery. Feature iteration goes through geozero, feeding the
//! same bounds accumulator every other source uses.

use std::collections::{HashMap, HashSet};

use flatgeobuf::{FallibleStreamingIterator, FgbReader};
use geozero::{ColumnValue, GeozeroGeometry, PropertyProcessor};

use crate::payload::{self, FeatureRef};
use crate::scan_core::{self, FeatureReadRequest, FeatureRecord, GeometryReadMode, ScanEntry};
use crate::wkb;
use crate::{
    BuildRequest, ConvertRequest, CoordinateDims, CrsInfo, DeclaredExtent, EdgeModel,
    EnvelopePolicy, GeoArtifact, GeoError, GeoIndex, GeoSource, GeometryEncoding,
    GeometryMetadataSource, GeometryProfile, GeometryScan, GeometrySelector, GeometryTypeSet,
    NullPolicy, PayloadPlan, PropertyProjection, RowBoundsSource, ScanRequest,
};

/// Name under which the (single, implicit) FlatGeobuf geometry is selectable.
const GEOMETRY_COLUMN: &str = "geometry";

/// Open a FlatGeobuf file.
///
/// Reads and snapshots the header only; features are streamed later by
/// [`FgbDataset::scan`], [`FgbDataset::build`], or [`FgbDataset::convert`],
/// which consume the reader (mirroring the Parquet
/// `GeoDataset`'s take-then-use pattern).
///
/// # Example
///
/// ```no_run
/// use std::fs::File;
/// use packed_spatial_index_geo::open_flatgeobuf;
///
/// let mut dataset = open_flatgeobuf(File::open("cities.fgb")?)?;
/// println!("{} features", dataset.profile()?.num_rows);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn open_flatgeobuf<R: std::io::Read + std::io::Seek>(
    reader: R,
) -> Result<FgbDataset<R>, GeoError> {
    let reader = FgbReader::open(reader).map_err(|e| GeoError::FlatGeobuf(e.to_string()))?;
    let header = HeaderInfo::from_header(&reader.header());
    let fingerprint = header.fingerprint();
    Ok(FgbDataset {
        reader: Some(reader),
        header,
        fingerprint,
    })
}

/// Owned snapshot of the FlatGeobuf header metadata.
#[derive(Debug, Clone)]
struct HeaderInfo {
    geometry_type: String,
    has_z: bool,
    has_m: bool,
    crs: CrsInfo,
    extent: Option<DeclaredExtent>,
    features_count: u64,
    columns: Vec<(String, String)>,
}

impl HeaderInfo {
    fn from_header(header: &flatgeobuf::Header<'_>) -> Self {
        let crs = match header.crs() {
            Some(crs) => {
                let code = crs.code();
                if code > 0 {
                    let org = crs.org().unwrap_or("EPSG");
                    CrsInfo::PresentString {
                        value: format!("{org}:{code}"),
                    }
                } else if let Some(code_string) = crs.code_string() {
                    CrsInfo::PresentString {
                        value: code_string.to_string(),
                    }
                } else if let Some(wkt) = crs.wkt() {
                    CrsInfo::PresentString {
                        value: wkt.to_string(),
                    }
                } else {
                    CrsInfo::Missing
                }
            }
            None => CrsInfo::Missing,
        };
        let extent = header.envelope().and_then(|envelope| {
            let values: Vec<f64> = envelope.iter().collect();
            (values.len() == 4 || values.len() == 6).then_some(DeclaredExtent { values })
        });
        let columns = header
            .columns()
            .map(|columns| {
                columns
                    .iter()
                    .map(|column| (column.name().to_string(), format!("{:?}", column.type_())))
                    .collect()
            })
            .unwrap_or_default();
        Self {
            geometry_type: format!("{:?}", header.geometry_type()),
            has_z: header.has_z(),
            has_m: header.has_m(),
            crs,
            extent,
            features_count: header.features_count(),
            columns,
        }
    }

    /// Canonical FNV-64 fingerprint over the header fields, mirroring the
    /// Parquet source fingerprint's shape (`fnv64:{hash:016x}`).
    fn fingerprint(&self) -> String {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        hash = payload::fnv(hash, self.geometry_type.as_bytes());
        hash = payload::fnv(hash, &[u8::from(self.has_z), u8::from(self.has_m)]);
        hash = payload::fnv(hash, format!("{:?}", self.crs).as_bytes());
        if let Some(extent) = &self.extent {
            for value in &extent.values {
                hash = payload::fnv(hash, &value.to_le_bytes());
            }
        }
        hash = payload::fnv(hash, &self.features_count.to_le_bytes());
        for (name, column_type) in &self.columns {
            hash = payload::fnv(hash, name.as_bytes());
            hash = payload::fnv(hash, column_type.as_bytes());
        }
        format!("fnv64:{hash:016x}")
    }

    fn coordinate_dims(&self) -> CoordinateDims {
        match (self.has_z, self.has_m) {
            (true, true) => CoordinateDims::Xyzm,
            (true, false) => CoordinateDims::Xyz,
            (false, true) => CoordinateDims::Xym,
            (false, false) => CoordinateDims::Xy,
        }
    }

    fn geometry_types(&self) -> GeometryTypeSet {
        if self.geometry_type == "Unknown" {
            return GeometryTypeSet::unknown();
        }
        let suffix = match (self.has_z, self.has_m) {
            (true, true) => " ZM",
            (true, false) => " Z",
            (false, true) => " M",
            (false, false) => "",
        };
        GeometryTypeSet {
            types: vec![format!("{}{suffix}", self.geometry_type)],
        }
    }
}

/// An opened FlatGeobuf file, ready to scan, build, or convert.
///
/// The header metadata stays available after the feature reader is consumed;
/// a second scan/build/convert call returns
/// [`GeoError::DatasetConsumed`] — open the file again for another pass.
pub struct FgbDataset<R> {
    reader: Option<FgbReader<R>>,
    header: HeaderInfo,
    fingerprint: String,
}

impl<R: std::io::Read + std::io::Seek> FgbDataset<R> {
    /// Profile of the (implicit) FlatGeobuf geometry column, from the header.
    ///
    /// Returns `Result` for signature parity with the other sources (see
    /// [`GeoSource::profile`](crate::GeoSource::profile)); FlatGeobuf profiling
    /// never fails.
    pub fn profile(&self) -> Result<GeometryProfile, GeoError> {
        Ok(GeometryProfile {
            column: GEOMETRY_COLUMN.to_string(),
            source: GeometryMetadataSource::FlatGeobuf,
            encoding: GeometryEncoding::FlatGeobuf,
            crs: self.header.crs.clone(),
            edges: EdgeModel::Planar,
            coordinate_dims: self.header.coordinate_dims(),
            geometry_types: self.header.geometry_types(),
            extent: self.header.extent.clone(),
            row_bounds: vec![RowBoundsSource::FeatureScan],
            num_rows: self.header.features_count,
        })
    }

    /// Stable fingerprint of the source header (FNV-64 over its fields).
    pub fn source_fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// Scan feature envelopes, feature references, and optional payloads.
    ///
    /// Mirrors `GeoDataset::scan`, including the
    /// projected-CRS guard for [`EnvelopePolicy::Geographic`]. Consumes the
    /// feature reader.
    pub fn scan(&mut self, req: ScanRequest) -> Result<GeometryScan, GeoError> {
        self.check_selector(&req.selector)?;
        if matches!(req.envelope, EnvelopePolicy::Geographic { .. })
            && self.header.crs.is_known_projected()
        {
            return Err(GeoError::Metadata(format!(
                "column `{GEOMETRY_COLUMN}` has a projected CRS; geographic antimeridian handling is only valid for lon/lat coordinates"
            )));
        }
        if let PayloadPlan::FeatureJson {
            properties: PropertyProjection::Include(include),
        } = &req.payload
        {
            for name in include {
                if !self.header.columns.iter().any(|(column, _)| column == name) {
                    return Err(GeoError::PropertyColumnNotFound(name.clone()));
                }
            }
        }
        let reader = self.reader.take().ok_or(GeoError::DatasetConsumed)?;
        let mut features = reader
            .select_all()
            .map_err(|e| GeoError::FlatGeobuf(e.to_string()))?;
        let collect_lons = matches!(req.envelope, EnvelopePolicy::Geographic { .. });
        let mut entries = scan_core::vec_with_u64_capacity_hint(self.header.features_count);
        let mut detected_dims = CoordinateDims::Unknown;
        let mut row = 0usize;
        loop {
            let feature = match features.next() {
                Ok(Some(feature)) => feature,
                Ok(None) => break,
                Err(e) => return Err(GeoError::FlatGeobuf(format!("features[{row}]: {e}"))),
            };
            let bounds =
                wkb::bounds_from_geozero(|processor| feature.process_geom(processor), collect_lons)
                    .map_err(|message| {
                        GeoError::FlatGeobuf(format!("features[{row}]: {message}"))
                    })?;
            let Some(bounds) = bounds else {
                match req.nulls {
                    NullPolicy::Skip => {
                        row += 1;
                        continue;
                    }
                    NullPolicy::Error => return Err(GeoError::NullGeometry { row }),
                }
            };
            detected_dims = detected_dims.merge(bounds.dims);
            let feature_ref = FeatureRef::row_number(row as u64);
            let payload_bytes = match &req.payload {
                PayloadPlan::None => None,
                PayloadPlan::RowRef => Some(payload::encode_feature_ref(&feature_ref)),
                PayloadPlan::RowWkb => {
                    use geozero::ToWkb;
                    let dims = geozero::CoordDimensions {
                        z: bounds.dims.has_z(),
                        m: bounds.dims.has_m(),
                        t: false,
                        tm: false,
                    };
                    let wkb_bytes = feature
                        .to_wkb(dims)
                        .map_err(|e| GeoError::FlatGeobuf(format!("features[{row}]: {e}")))?;
                    Some(payload::encode_feature_wkb(&feature_ref, &wkb_bytes))
                }
                PayloadPlan::FeatureJson { properties } => {
                    use geozero::ToJson;
                    let geometry_text = feature
                        .to_json()
                        .map_err(|e| GeoError::FlatGeobuf(format!("features[{row}]: {e}")))?;
                    let geometry: serde_json::Value = serde_json::from_str(&geometry_text)
                        .map_err(|e| GeoError::FlatGeobuf(format!("features[{row}]: {e}")))?;
                    let projected = collect_properties(feature, properties)
                        .map_err(|e| GeoError::FlatGeobuf(format!("features[{row}]: {e}")))?;
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
            row += 1;
        }
        let mut profile = self.profile()?;
        // A features_count of 0 means "unknown" in the FlatGeobuf header;
        // after a full pass the real count is known.
        if profile.num_rows == 0 {
            profile.num_rows = row as u64;
        }
        scan_core::assemble_scan(entries, &req, profile, detected_dims)
    }

    /// Build an in-memory [`GeoIndex`] over the file's features.
    ///
    /// Mirrors `GeoDataset::build`. Consumes the
    /// feature reader.
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

    /// Convert the file into a streamable `PSINDEX` buffer.
    ///
    /// Mirrors `GeoDataset::convert_into`;
    /// the artifact manifest records `source_format: "flatgeobuf"`. Consumes
    /// the feature reader.
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

    /// Convert the file into a new `Vec<u8>`.
    ///
    /// Convenience wrapper around [`FgbDataset::convert_into`].
    pub fn convert(&mut self, req: ConvertRequest) -> Result<Vec<u8>, GeoError> {
        let mut out = Vec::new();
        self.convert_into(req, &mut out)?;
        Ok(out)
    }

    /// Read source features back by [`FeatureRef`](crate::FeatureRef).
    ///
    /// This streams the FlatGeobuf feature section once and consumes the
    /// dataset reader, just like [`scan`](Self::scan).
    pub fn read_features(
        &mut self,
        req: FeatureReadRequest,
    ) -> Result<Vec<FeatureRecord>, GeoError> {
        self.check_selector(&req.selector)?;
        self.check_expected_fingerprint(req.expected_source_fingerprint.as_ref())?;
        self.check_property_projection(&req.properties)?;
        let output = scan_core::ordered_feature_refs(
            &req.features,
            (self.header.features_count > 0).then_some(self.header.features_count),
            req.order,
            req.duplicates,
        )?;
        let wanted: HashSet<u64> = output.iter().map(|feature| feature.row_number).collect();
        let reader = self.reader.take().ok_or(GeoError::DatasetConsumed)?;
        let mut features = reader
            .select_all()
            .map_err(|e| GeoError::FlatGeobuf(e.to_string()))?;
        let mut materialized = HashMap::new();
        let mut row = 0u64;
        loop {
            let feature = match features.next() {
                Ok(Some(feature)) => feature,
                Ok(None) => break,
                Err(e) => return Err(GeoError::FlatGeobuf(format!("features[{row}]: {e}"))),
            };
            if wanted.contains(&row) {
                materialized.insert(
                    row,
                    materialize_feature(feature, req.geometry, &req.properties, row as usize)?,
                );
            }
            row += 1;
        }
        output
            .into_iter()
            .map(|feature| {
                let Some(record) = materialized.get(&feature.row_number) else {
                    return Err(GeoError::FeatureRowOutOfBounds {
                        row_number: feature.row_number,
                        num_rows: row,
                    });
                };
                Ok(FeatureRecord {
                    feature,
                    geometry_wkb: record.geometry_wkb.clone(),
                    geometry_json: record.geometry_json.clone(),
                    properties: record.properties.clone(),
                })
            })
            .collect()
    }

    fn check_selector(&self, selector: &GeometrySelector) -> Result<(), GeoError> {
        match selector {
            GeometrySelector::Default | GeometrySelector::FirstUsable => Ok(()),
            GeometrySelector::Name(name) if name == GEOMETRY_COLUMN => Ok(()),
            GeometrySelector::Name(name) => Err(GeoError::GeometryColumnNotFound(name.clone())),
            GeometrySelector::GeoParquetPrimary | GeometrySelector::SingleNativeParquet => {
                Err(GeoError::Metadata(
                    "selector applies to Parquet sources; use Default or Name(\"geometry\") for FlatGeobuf".to_string(),
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
            if !self.header.columns.iter().any(|(column, _)| column == name) {
                return Err(GeoError::PropertyColumnNotFound(name.clone()));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct MaterializedFgbFeature {
    geometry_wkb: Option<Vec<u8>>,
    geometry_json: Option<serde_json::Value>,
    properties: serde_json::Value,
}

fn materialize_feature(
    feature: &flatgeobuf::FgbFeature,
    geometry: GeometryReadMode,
    properties: &PropertyProjection,
    row: usize,
) -> Result<MaterializedFgbFeature, GeoError> {
    let bounds = wkb::bounds_from_geozero(|processor| feature.process_geom(processor), false)
        .map_err(|message| GeoError::FlatGeobuf(format!("features[{row}]: {message}")))?;
    let geometry_wkb = match (geometry, bounds.as_ref()) {
        (GeometryReadMode::Omit, _) | (_, None) => None,
        (GeometryReadMode::Wkb, Some(bounds)) => Some(feature_wkb(feature, bounds, row)?),
    };
    let geometry_json = if bounds.is_some() {
        Some(feature_geometry_json(feature, row)?)
    } else {
        None
    };
    let properties = collect_properties(feature, properties)
        .map_err(|e| GeoError::FlatGeobuf(format!("features[{row}]: {e}")))?;
    Ok(MaterializedFgbFeature {
        geometry_wkb,
        geometry_json,
        properties: serde_json::Value::Object(properties),
    })
}

fn feature_wkb(
    feature: &flatgeobuf::FgbFeature,
    bounds: &wkb::GeometryBounds,
    row: usize,
) -> Result<Vec<u8>, GeoError> {
    use geozero::ToWkb;
    let dims = geozero::CoordDimensions {
        z: bounds.dims.has_z(),
        m: bounds.dims.has_m(),
        t: false,
        tm: false,
    };
    feature
        .to_wkb(dims)
        .map_err(|e| GeoError::FlatGeobuf(format!("features[{row}]: {e}")))
}

fn feature_geometry_json(
    feature: &flatgeobuf::FgbFeature,
    row: usize,
) -> Result<serde_json::Value, GeoError> {
    use geozero::ToJson;
    let geometry_text = feature
        .to_json()
        .map_err(|e| GeoError::FlatGeobuf(format!("features[{row}]: {e}")))?;
    serde_json::from_str(&geometry_text)
        .map_err(|e| GeoError::FlatGeobuf(format!("features[{row}]: {e}")))
}

/// Collect a feature's properties into a typed JSON object, applying the
/// projection. Numbers stay numbers, `Json` columns are parsed, binary is
/// base64, datetimes stay ISO8601 strings — no lossy stringification.
fn collect_properties(
    feature: &flatgeobuf::FgbFeature,
    projection: &PropertyProjection,
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    if matches!(projection, PropertyProjection::None) {
        return Ok(serde_json::Map::new());
    }
    struct Collector<'a> {
        projection: &'a PropertyProjection,
        out: serde_json::Map<String, serde_json::Value>,
        error: Option<String>,
    }
    impl PropertyProcessor for Collector<'_> {
        fn property(
            &mut self,
            _idx: usize,
            name: &str,
            value: &ColumnValue,
        ) -> geozero::error::Result<bool> {
            let keep = match self.projection {
                PropertyProjection::None => false,
                PropertyProjection::AllNonGeometry => true,
                PropertyProjection::Include(include) => include.iter().any(|n| n == name),
                PropertyProjection::Exclude(exclude) => !exclude.iter().any(|n| n == name),
            };
            if keep {
                match json_value(value) {
                    Ok(json) => {
                        self.out.insert(name.to_string(), json);
                    }
                    Err(message) => {
                        self.error = Some(format!("property `{name}`: {message}"));
                        return Ok(true);
                    }
                }
            }
            Ok(false)
        }
    }
    let mut collector = Collector {
        projection,
        out: serde_json::Map::new(),
        error: None,
    };
    use geozero::FeatureProperties;
    feature
        .process_properties(&mut collector)
        .map_err(|e| e.to_string())?;
    match collector.error {
        Some(message) => Err(message),
        None => Ok(collector.out),
    }
}

fn json_value(value: &ColumnValue) -> Result<serde_json::Value, String> {
    use base64::Engine;
    Ok(match value {
        ColumnValue::Byte(v) => serde_json::Value::from(*v),
        ColumnValue::UByte(v) => serde_json::Value::from(*v),
        ColumnValue::Bool(v) => serde_json::Value::from(*v),
        ColumnValue::Short(v) => serde_json::Value::from(*v),
        ColumnValue::UShort(v) => serde_json::Value::from(*v),
        ColumnValue::Int(v) => serde_json::Value::from(*v),
        ColumnValue::UInt(v) => serde_json::Value::from(*v),
        ColumnValue::Long(v) => serde_json::Value::from(*v),
        ColumnValue::ULong(v) => serde_json::Value::from(*v),
        ColumnValue::Float(v) => serde_json::Value::from(*v),
        ColumnValue::Double(v) => serde_json::Value::from(*v),
        ColumnValue::String(v) => serde_json::Value::from(*v),
        ColumnValue::Json(v) => {
            serde_json::from_str(v).map_err(|e| format!("invalid JSON column: {e}"))?
        }
        ColumnValue::DateTime(v) => serde_json::Value::from(*v),
        ColumnValue::Binary(v) => {
            serde_json::Value::from(base64::engine::general_purpose::STANDARD.encode(v))
        }
    })
}

impl<R: std::io::Read + std::io::Seek> GeoSource for FgbDataset<R> {
    fn profile(&self) -> Result<GeometryProfile, GeoError> {
        FgbDataset::profile(self)
    }

    fn source_fingerprint(&self) -> &str {
        FgbDataset::source_fingerprint(self)
    }

    fn scan(&mut self, req: ScanRequest) -> Result<GeometryScan, GeoError> {
        FgbDataset::scan(self, req)
    }

    fn build(&mut self, req: BuildRequest) -> Result<GeoIndex, GeoError> {
        FgbDataset::build(self, req)
    }

    fn convert(&mut self, req: ConvertRequest) -> Result<Vec<u8>, GeoError> {
        FgbDataset::convert(self, req)
    }

    fn convert_into(
        &mut self,
        req: ConvertRequest,
        out: &mut Vec<u8>,
    ) -> Result<GeoArtifact, GeoError> {
        FgbDataset::convert_into(self, req, out)
    }
}
