use std::io;

/// A source of bytes addressable by absolute offset.
///
/// This is the only capability [`StreamIndex2D`](super::StreamIndex2D) needs
/// from its backing store, so a local file, an in-memory slice, or a remote
/// object behind HTTP range requests can all drive the same streaming queries.
///
/// Implementations must read from an absolute offset **without** disturbing any
/// shared cursor (hence `&self`, not `&mut self`), so one reader can serve
/// concurrent queries safely.
///
/// # A remote (HTTP range) reader
///
/// Implement the single required method to query an index that lives in object
/// storage — no crate dependency on any HTTP client:
///
/// ```ignore
/// use std::io;
/// use packed_spatial_index::RangeReader;
///
/// struct HttpRange {
///     url: String,
///     client: reqwest::blocking::Client,
/// }
///
/// impl RangeReader for HttpRange {
///     fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
///         let end = offset + buf.len() as u64 - 1;
///         let bytes = self
///             .client
///             .get(&self.url)
///             .header("Range", format!("bytes={offset}-{end}"))
///             .send()
///             .and_then(|r| r.error_for_status())
///             .and_then(|r| r.bytes())
///             .map_err(io::Error::other)?;
///         if bytes.len() != buf.len() {
///             return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short range"));
///         }
///         buf.copy_from_slice(&bytes);
///         Ok(())
///     }
///     // `len` defaults to `None`; `open` then skips the length cross-check and
///     // relies on reads past the end failing. Override it (e.g. from a HEAD
///     // request) for a stricter check.
/// }
/// ```
// `len` reports the source's total byte length if known; "emptiness" is not a
// meaningful concept for a random-access byte source, so no `is_empty`.
#[allow(clippy::len_without_is_empty)]
pub trait RangeReader {
    /// Read exactly `buf.len()` bytes starting at byte `offset`, filling `buf`.
    ///
    /// Returns an [`io::ErrorKind::UnexpectedEof`] error if fewer bytes are
    /// available. A zero-length `buf` always succeeds.
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()>;

    /// Total length in bytes, if known.
    ///
    /// Local files report their size; a remote reader may return [`None`], in
    /// which case [`StreamIndex2D::open`](super::StreamIndex2D::open) skips the
    /// exact-length cross-check and instead relies on reads past the end failing.
    fn len(&self) -> Option<u64> {
        None
    }
}

pub(super) fn unexpected_eof() -> io::Error {
    io::Error::new(
        io::ErrorKind::UnexpectedEof,
        "read past the end of the range source",
    )
}

/// A [`RangeReader`] over an in-memory byte buffer (`&[u8]`, `Vec<u8>`, a memory
/// map, ...). Reads are bounds-checked copies out of the buffer.
pub struct SliceReader<T> {
    data: T,
}

impl<T: AsRef<[u8]>> SliceReader<T> {
    /// Wrap an in-memory buffer.
    pub fn new(data: T) -> Self {
        Self { data }
    }

    /// Return the wrapped buffer.
    pub fn into_inner(self) -> T {
        self.data
    }
}

impl<T: AsRef<[u8]>> RangeReader for SliceReader<T> {
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let data = self.data.as_ref();
        let start = usize::try_from(offset).map_err(|_| unexpected_eof())?;
        let end = start.checked_add(buf.len()).ok_or_else(unexpected_eof)?;
        let src = data.get(start..end).ok_or_else(unexpected_eof)?;
        buf.copy_from_slice(src);
        Ok(())
    }

    fn len(&self) -> Option<u64> {
        Some(self.data.as_ref().len() as u64)
    }
}

/// A [`RangeReader`] over a local file using positioned reads.
///
/// Positioned reads (`pread` on Unix, `seek_read` on Windows) don't move a
/// shared cursor, so the reader takes `&self` and one open file can serve many
/// concurrent queries. Available on Unix and Windows; other targets can
/// implement [`RangeReader`] directly.
#[cfg(any(unix, windows))]
pub struct FileReader {
    file: std::fs::File,
    len: u64,
}

#[cfg(any(unix, windows))]
impl FileReader {
    /// Open a file at `path` for streaming reads.
    pub fn open(path: impl AsRef<std::path::Path>) -> io::Result<Self> {
        Self::from_file(std::fs::File::open(path)?)
    }

    /// Wrap an already-open file. Its length is queried once via metadata.
    pub fn from_file(file: std::fs::File) -> io::Result<Self> {
        let len = file.metadata()?.len();
        Ok(Self { file, len })
    }
}

#[cfg(any(unix, windows))]
impl RangeReader for FileReader {
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        #[cfg(unix)]
        {
            std::os::unix::fs::FileExt::read_exact_at(&self.file, buf, offset)
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::FileExt;
            let mut filled = 0usize;
            while filled < buf.len() {
                let n = self
                    .file
                    .seek_read(&mut buf[filled..], offset + filled as u64)?;
                if n == 0 {
                    return Err(unexpected_eof());
                }
                filled += n;
            }
            Ok(())
        }
    }

    fn len(&self) -> Option<u64> {
        Some(self.len)
    }
}
