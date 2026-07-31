//! Query-shape decision logic shared by every artifact query frontend
//! (the native server, the Cloudflare Worker demo, and any other consumer).
//!
//! These functions are pure and I/O-free: given a manifest and the requested
//! shape, they decide what a query response should contain. Keeping one copy
//! here means a rule like [`resolve_identity`] cannot drift between two
//! frontends the way it once did.

use crate::{GeoArtifactManifest, PayloadPlan};

/// Payload materialization mode for a search query.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PayloadMode {
    /// Omit the payload object from each match.
    None,
    /// Return payload kind and cheap metadata only.
    #[default]
    Summary,
    /// Return full payload values where the artifact stores them.
    Full,
}

impl PayloadMode {
    /// Wire-vocabulary string, for callers building JSON by hand instead of
    /// through [`serde::Serialize`].
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Summary => "summary",
            Self::Full => "full",
        }
    }
}

/// Source-identity detail returned for each match.
///
/// Index entries carry a fixed-width feature reference, but a source
/// `featureId` lives inside the payload body, so returning one costs a body
/// read. This mode makes that cost explicit instead of letting it depend on
/// which internal path answered the query.
///
/// It is resolved against the collection and the rest of the request rather
/// than taken literally — see [`resolve_identity`] — so responses echo the
/// mode that applied.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentityMode {
    /// Return the fixed-width feature reference only; omit `featureId`.
    #[default]
    Ref,
    /// Read payload bodies for the returned page so `featureId` is included.
    Full,
}

impl IdentityMode {
    /// Wire-vocabulary string, for callers building JSON by hand instead of
    /// through [`serde::Serialize`].
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ref => "ref",
            Self::Full => "full",
        }
    }
}

/// Result granularity for a search query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResultLevel {
    /// One record per source feature; split index entries are deduplicated.
    Feature,
    /// One record per index entry, including split parts.
    Entry,
}

impl ResultLevel {
    /// Wire-vocabulary string, for callers building JSON by hand instead of
    /// through [`serde::Serialize`].
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Feature => "feature",
            Self::Entry => "entry",
        }
    }
}

/// `level=feature` was requested but the artifact stores no feature
/// references to group entries by.
///
/// Carries no data of its own: a frontend that has more context than the
/// artifact usually words this its own way — the native server names the
/// collection id, the Worker demo does not — so [`resolve_level`] classifies
/// the failure and leaves the wording to the caller. The [`Display`] text
/// below is what a caller with nothing to add can say.
///
/// [`Display`]: std::fmt::Display
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum LevelError {
    /// The artifact's payload plan stores no feature references.
    #[error("the artifact stores no feature references; ask for entry level")]
    FeatureRefsUnavailable,
}

/// Resolve `level`, which the artifact's payload plan has a say in.
///
/// The default cannot be a constant: an artifact with no payload stores no
/// feature references, so there is nothing to group entries into, and asking
/// for feature level there is a request the artifact cannot serve rather than
/// one to quietly reinterpret.
pub fn resolve_level(
    requested: Option<ResultLevel>,
    plan: &PayloadPlan,
) -> Result<ResultLevel, LevelError> {
    let has_feature_refs = !matches!(plan, PayloadPlan::None);
    match requested {
        None => Ok(if has_feature_refs {
            ResultLevel::Feature
        } else {
            ResultLevel::Entry
        }),
        Some(ResultLevel::Feature) if !has_feature_refs => Err(LevelError::FeatureRefsUnavailable),
        Some(level) => Ok(level),
    }
}

/// Whether this artifact can produce a source feature id at all.
///
/// The payload plan answers a different question — `FeatureJson` says where
/// an id would live, not whether one exists — and only a GeoJSON source ever
/// supplies one, so the manifest has to say. An artifact written before it
/// recorded that says nothing, and the benefit of the doubt goes to the
/// caller: assume ids are there until an artifact states otherwise.
pub fn stores_feature_ids(manifest: &GeoArtifactManifest) -> bool {
    matches!(manifest.payload_plan, PayloadPlan::FeatureJson { .. })
        && manifest.stores_feature_ids.unwrap_or(true)
}

/// Pick the identity mode, which the artifact and the rest of the request
/// both have a say in.
///
/// `full` costs a page of body reads, so it must not be granted where it buys
/// nothing and must not be withheld where it costs nothing:
///
/// - an artifact storing no source id resolves to `ref` whatever was asked.
///   Refusing instead would follow the `level` rule, but `level` changes what
///   a record *is*, while `identity` only adds one optional field — and the
///   echoed mode already tells the caller which way it went.
/// - `payload=full` reads the bodies regardless, and the id comes back in the
///   returned GeoJSON `id` member anyway. Withholding it from `featureRef`
///   there hides nothing and costs the caller a second parameter.
pub fn resolve_identity(
    stores_feature_ids: bool,
    payload: PayloadMode,
    requested: IdentityMode,
) -> IdentityMode {
    if !stores_feature_ids {
        IdentityMode::Ref
    } else if payload == PayloadMode::Full {
        IdentityMode::Full
    } else {
        requested
    }
}

/// Whether the requested shape needs payload bodies for the returned page.
///
/// A body read serves two different needs: the payload value itself, and the
/// source `featureId`. `identity` here is the mode that survived
/// [`resolve_identity`], which is `Full` only where an id can actually be
/// recovered — so this does not have to re-examine the payload plan to avoid
/// buying a page of I/O for a byte-identical answer.
pub fn needs_payload_bodies(payload: PayloadMode, identity: IdentityMode) -> bool {
    payload == PayloadMode::Full || identity == IdentityMode::Full
}

/// Strip the internal `feature_ref` member a `FeatureJson` payload carries
/// before it reaches a client.
pub fn public_feature_json(mut feature: serde_json::Value) -> serde_json::Value {
    if let Some(object) = feature.as_object_mut() {
        object.remove("feature_ref");
    }
    feature
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AntimeridianPolicy, CoordinateDims, CrsInfo, EdgeModel, GeometryEncoding, NullPolicy,
        PropertyProjection, StoragePrecision,
    };

    fn manifest_with(
        payload_plan: PayloadPlan,
        stores_feature_ids: Option<bool>,
    ) -> GeoArtifactManifest {
        GeoArtifactManifest {
            schema_version: 2,
            source_format: "geojson".to_string(),
            source_fingerprint: String::new(),
            selected_column: "geometry".to_string(),
            crs: CrsInfo::ImpliedDefault {
                value: "OGC:CRS84".to_string(),
            },
            edges: EdgeModel::Planar,
            encoding: GeometryEncoding::GeoJson,
            dims: CoordinateDims::Xy,
            storage_precision: StoragePrecision::F64,
            null_policy: NullPolicy::Skip,
            antimeridian_policy: AntimeridianPolicy::Reject,
            payload_plan,
            feature_count: 0,
            index_entry_count: 0,
            entries_may_duplicate_rows: false,
            stores_feature_ids,
        }
    }

    fn feature_json_plan() -> PayloadPlan {
        PayloadPlan::FeatureJson {
            properties: PropertyProjection::AllNonGeometry,
        }
    }

    #[test]
    fn only_full_identity_reads_payload_bodies() {
        assert!(needs_payload_bodies(PayloadMode::Full, IdentityMode::Ref));
        assert!(needs_payload_bodies(
            PayloadMode::Summary,
            IdentityMode::Full
        ));
        assert!(!needs_payload_bodies(
            PayloadMode::Summary,
            IdentityMode::Ref
        ));
        assert!(!needs_payload_bodies(PayloadMode::None, IdentityMode::Ref));
    }

    /// A collection with no source id to give must not sell one: `full` there
    /// would buy a page of body reads for a byte-identical answer.
    #[test]
    fn identity_resolves_down_where_no_id_is_stored() {
        for payload in [PayloadMode::None, PayloadMode::Summary, PayloadMode::Full] {
            for requested in [IdentityMode::Ref, IdentityMode::Full] {
                assert_eq!(
                    resolve_identity(false, payload, requested),
                    IdentityMode::Ref,
                    "{payload:?} {requested:?}"
                );
            }
        }
    }

    /// `payload=full` reads the bodies anyway and returns the id inside the
    /// GeoJSON feature, so withholding it from `featureRef` hides nothing.
    #[test]
    fn identity_resolves_up_where_the_bodies_are_read_anyway() {
        assert_eq!(
            resolve_identity(true, PayloadMode::Full, IdentityMode::Ref),
            IdentityMode::Full
        );
        for payload in [PayloadMode::None, PayloadMode::Summary] {
            assert_eq!(
                resolve_identity(true, payload, IdentityMode::Ref),
                IdentityMode::Ref,
                "{payload:?}"
            );
            assert_eq!(
                resolve_identity(true, payload, IdentityMode::Full),
                IdentityMode::Full,
                "{payload:?}"
            );
        }
    }

    /// The plan says where an id would live; the manifest flag says whether
    /// one is there. An artifact predating the flag keeps the benefit of the
    /// doubt, because its bodies may well carry ids.
    #[test]
    fn only_a_feature_json_artifact_that_kept_ids_stores_them() {
        let plans = [
            (feature_json_plan(), [false, true, true]),
            (PayloadPlan::RowWkb, [false, false, false]),
            (PayloadPlan::RowRef, [false, false, false]),
            (PayloadPlan::None, [false, false, false]),
        ];
        for (plan, expected) in plans {
            for (flag, expected) in [Some(false), Some(true), None].into_iter().zip(expected) {
                let manifest = manifest_with(plan.clone(), flag);
                assert_eq!(stores_feature_ids(&manifest), expected, "{plan:?} {flag:?}");
            }
        }
    }

    #[test]
    fn level_resolution_follows_the_payload_plan() {
        let json = feature_json_plan();
        // Unspecified: feature level where refs exist, entry level where they
        // cannot. A constant default cannot express that.
        assert_eq!(resolve_level(None, &json).unwrap(), ResultLevel::Feature);
        assert_eq!(
            resolve_level(None, &PayloadPlan::None).unwrap(),
            ResultLevel::Entry
        );
        // Explicit entry level is always available.
        assert_eq!(
            resolve_level(Some(ResultLevel::Entry), &PayloadPlan::None).unwrap(),
            ResultLevel::Entry
        );
        // Explicit feature level on a payload-less artifact is a request the
        // artifact cannot serve, not one to quietly reinterpret.
        assert_eq!(
            resolve_level(Some(ResultLevel::Feature), &PayloadPlan::None),
            Err(LevelError::FeatureRefsUnavailable)
        );
    }

    #[test]
    fn public_feature_json_strips_the_internal_feature_ref() {
        let feature = serde_json::json!({
            "type": "Feature",
            "id": "f1",
            "feature_ref": {"rowNumber": 1},
            "properties": {"name": "a"},
        });
        let public = public_feature_json(feature);
        assert!(public.as_object().unwrap().get("feature_ref").is_none());
        assert_eq!(public["id"], "f1");
        assert_eq!(public["properties"]["name"], "a");
    }

    /// A caller that adds no context of its own still has to be able to
    /// report this: propagate it with `?`, box it, or print it.
    #[test]
    fn level_error_reports_itself_like_any_other_error() {
        fn propagates() -> Result<ResultLevel, Box<dyn std::error::Error>> {
            Ok(resolve_level(
                Some(ResultLevel::Feature),
                &PayloadPlan::None,
            )?)
        }

        let boxed = propagates().unwrap_err();
        assert_eq!(
            boxed.to_string(),
            "the artifact stores no feature references; ask for entry level"
        );
    }
}
