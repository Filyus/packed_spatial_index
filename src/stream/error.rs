use std::io;

use crate::persistence::LoadError;

/// Error returned by the streaming reader.
#[derive(Debug)]
pub enum StreamError {
    /// An I/O error from the backing [`RangeReader`](super::RangeReader).
    Io(io::Error),
    /// The bytes are not a valid index of the expected variant. Carries the same
    /// [`LoadError`] categories as the in-memory loader.
    Format(LoadError),
    /// Payloads were requested but the index has no payload section.
    NoPayload,
    /// The query exceeded a configured [`StreamLimits`](super::StreamLimits)
    /// budget and was aborted.
    LimitExceeded,
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamError::Io(err) => write!(f, "streaming read failed: {err}"),
            StreamError::Format(err) => write!(f, "{err}"),
            StreamError::NoPayload => write!(f, "index has no payload section"),
            StreamError::LimitExceeded => write!(f, "query exceeded its configured limits"),
        }
    }
}

impl std::error::Error for StreamError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StreamError::Io(err) => Some(err),
            StreamError::Format(err) => Some(err),
            StreamError::NoPayload | StreamError::LimitExceeded => None,
        }
    }
}

impl From<io::Error> for StreamError {
    fn from(err: io::Error) -> Self {
        StreamError::Io(err)
    }
}

impl From<LoadError> for StreamError {
    fn from(err: LoadError) -> Self {
        StreamError::Format(err)
    }
}
