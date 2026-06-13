//! Query a serialized index straight off disk with the `stream` feature, without
//! loading the whole file into an owned index.
//!
//! Run with: `cargo run --example stream_query --features stream`

use packed_spatial_index::{Box2D, FileReader, Index2DBuilder, StreamIndex2D};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a small index and serialize it to a temporary file.
    let mut builder = Index2DBuilder::new(5);
    builder.add(Box2D::new(0.0, 0.0, 1.0, 1.0));
    builder.add(Box2D::new(2.0, 2.0, 3.0, 3.0));
    builder.add(Box2D::new(5.0, 5.0, 6.0, 6.0));
    builder.add(Box2D::new(8.0, 8.0, 9.0, 9.0));
    builder.add(Box2D::new(0.5, 0.5, 2.5, 2.5));
    let index = builder.finish()?;

    let path = std::env::temp_dir().join("psi_stream_example.psindex");
    std::fs::write(&path, index.to_bytes())?;

    // Open over a positioned-read FileReader and query it. The reader fetches
    // only the tree nodes the query touches — for this tiny index that is the
    // whole thing, but the same code scales to files larger than memory.
    let stream = StreamIndex2D::open(FileReader::open(&path)?)?;

    let query = Box2D::new(0.0, 0.0, 3.0, 3.0);
    let mut hits = stream.search(query)?;
    hits.sort_unstable();

    println!(
        "streamed {} items; hits for {query:?}: {hits:?}",
        stream.num_items()
    );
    assert_eq!(hits, vec![0, 1, 4]);

    std::fs::remove_file(&path).ok();
    Ok(())
}
