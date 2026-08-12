//! Writes valid artifacts into `fuzz/corpus/load`, so a run starts inside the
//! chunk table instead of spending its first minutes rediscovering the magic.
//!
//! ```text
//! cargo run --bin seed          # from fuzz/
//! ```
//!
//! Deliberately several shapes rather than one: a tree with a single level, one
//! with several, an empty index, and the SIMD and 3D layouts, because the paths
//! that validation has to tell apart are the ones a mutated seed will blur.

use std::fs;
use std::path::Path;

use packed_spatial_index::{Box2D, Box3D, Index2DBuilder, Index3DBuilder};

fn boxes_2d(n: usize) -> Vec<Box2D> {
    (0..n)
        .map(|i| {
            let x = (i % 97) as f64 * 3.0;
            let y = (i / 97) as f64 * 3.0;
            Box2D::new(x, y, x + 2.0, y + 2.0)
        })
        .collect()
}

fn write(dir: &Path, name: &str, bytes: &[u8]) {
    let path = dir.join(name);
    fs::write(&path, bytes).expect("corpus directory is writable");
    println!("{} ({} bytes)", path.display(), bytes.len());
}

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus/load");
    fs::create_dir_all(&dir).expect("can create the corpus directory");

    for &n in &[0usize, 1, 14, 100, 5_000] {
        let mut builder = Index2DBuilder::new(n);
        for b in boxes_2d(n) {
            builder.add(b);
        }
        let index = builder.finish().expect("builder accepts its own boxes");
        write(&dir, &format!("index2d_{n}"), &index.to_bytes());

        let mut builder = Index2DBuilder::new(n).node_size(4);
        for b in boxes_2d(n) {
            builder.add(b);
        }
        let simd = builder.finish_simd().expect("builder accepts its own boxes");
        write(&dir, &format!("simd2d_n4_{n}"), &simd.to_bytes());

        let mut builder = Index3DBuilder::new(n);
        for (i, b) in boxes_2d(n).into_iter().enumerate() {
            let z = (i % 13) as f64;
            builder.add(Box3D::new(b.min_x, b.min_y, z, b.max_x, b.max_y, z + 1.0));
        }
        let index = builder.finish().expect("builder accepts its own boxes");
        write(&dir, &format!("index3d_{n}"), &index.to_bytes());
    }
}
