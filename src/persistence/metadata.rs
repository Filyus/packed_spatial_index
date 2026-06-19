use super::container::{find_chunk, parse_container};
use super::{ByteWriter, LoadError, TAG_TREE, read_u16_at, read_u32_at};

/// Optional descriptive metadata chunk (CRS / content type / attribution).
/// Optional — readers that do not care skip it.
pub(crate) const TAG_META: [u8; 4] = *b"META";

// `META` field ids. The chunk is a flat list of `(id: u16, len: u32, bytes)`
// fields read until the chunk ends; an unknown id is skipped, so new fields are
// non-breaking. Values are opaque UTF-8 strings the writer supplied.
const META_CRS: u16 = 0;
const META_CONTENT_TYPE: u16 = 1;
const META_ATTRIBUTION: u16 = 2;

/// Descriptive fields to write into a `META` chunk (borrowed, write side).
#[derive(Default)]
pub(crate) struct MetaFields<'a> {
    pub(crate) crs: Option<&'a str>,
    pub(crate) content_type: Option<&'a str>,
    pub(crate) attribution: Option<&'a str>,
}

impl MetaFields<'_> {
    pub(crate) fn is_empty(&self) -> bool {
        self.crs.is_none() && self.content_type.is_none() && self.attribution.is_none()
    }

    /// Byte length of the `META` chunk content for these fields.
    pub(crate) fn content_len(&self) -> usize {
        [self.crs, self.content_type, self.attribution]
            .into_iter()
            .flatten()
            .map(|s| 6 + s.len()) // id(2) + len(4) + bytes
            .sum()
    }
}

/// Descriptive metadata read from a file's `META` chunk. Every field is an opaque
/// string the writer supplied; this crate does not interpret them (e.g. the CRS
/// is whatever identifier the producer chose, such as `"EPSG:4326"`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FileMetadata {
    /// Coordinate reference system identifier, if present.
    pub crs: Option<String>,
    /// Payload content type (media type), if present.
    pub content_type: Option<String>,
    /// Attribution / license string, if present.
    pub attribution: Option<String>,
}

impl ByteWriter<'_> {
    pub(crate) fn write_meta(&mut self, fields: &MetaFields<'_>) {
        for (id, value) in [
            (META_CRS, fields.crs),
            (META_CONTENT_TYPE, fields.content_type),
            (META_ATTRIBUTION, fields.attribution),
        ] {
            if let Some(s) = value {
                self.write_u16(id);
                self.write_u32(s.len() as u32);
                self.write_bytes(s.as_bytes());
            }
        }
    }
}

/// Read the optional descriptive metadata from a serialized index, without
/// loading the index. Returns an empty [`FileMetadata`] when there is no `META`
/// chunk.
pub fn read_metadata(bytes: &[u8]) -> Result<FileMetadata, LoadError> {
    let chunks = parse_container(bytes, &[TAG_TREE])?;
    match find_chunk(&chunks, TAG_META) {
        Some(m) => parse_meta(&bytes[m.offset..m.offset + m.len]),
        None => Ok(FileMetadata::default()),
    }
}

/// Parse a `META` chunk's flat field list into owned strings.
fn parse_meta(content: &[u8]) -> Result<FileMetadata, LoadError> {
    let mut md = FileMetadata::default();
    let mut off = 0;
    while off < content.len() {
        let id = read_u16_at(content, off)?;
        let len = read_u32_at(content, off + 2)? as usize;
        let start = off + 6;
        let end = start.checked_add(len).ok_or(LoadError::IntegerOverflow)?;
        let bytes = content.get(start..end).ok_or(LoadError::Truncated)?;
        let s = std::str::from_utf8(bytes).map_err(|_| LoadError::InvalidTree)?;
        match id {
            META_CRS => md.crs = Some(s.to_owned()),
            META_CONTENT_TYPE => md.content_type = Some(s.to_owned()),
            META_ATTRIBUTION => md.attribution = Some(s.to_owned()),
            _ => {} // unknown field: skip
        }
        off = end;
    }
    Ok(md)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_parses_known_fields_and_skips_unknown() {
        // A META chunk content with crs(0), an unknown future field(99), and
        // attribution(2). The unknown field must be skipped, not break parsing.
        let mut content = Vec::new();
        let put = |c: &mut Vec<u8>, id: u16, value: &[u8]| {
            c.extend_from_slice(&id.to_le_bytes());
            c.extend_from_slice(&(value.len() as u32).to_le_bytes());
            c.extend_from_slice(value);
        };
        put(&mut content, 0, b"EPSG:4326"); // crs
        put(&mut content, 99, b"from-the-future"); // unknown -> skipped
        put(&mut content, 2, b"attribution-text"); // attribution

        let md = parse_meta(&content).unwrap();
        assert_eq!(md.crs.as_deref(), Some("EPSG:4326"));
        assert_eq!(md.attribution.as_deref(), Some("attribution-text"));
        assert_eq!(md.content_type, None);

        // Empty content -> all fields absent.
        assert_eq!(parse_meta(&[]).unwrap(), FileMetadata::default());
    }
}
