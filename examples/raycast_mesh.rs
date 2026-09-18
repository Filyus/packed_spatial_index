//! Streamable bounding-volume hierarchy over a triangle mesh.
//!
//! Build an index over each triangle's bounding box, attach the triangles as a
//! fixed-width payload (no offset table), then serialize. The same bytes load
//! back zero-copy: a ray query narrows to candidate triangles through the index
//! (broad phase), and an exact ray-triangle test runs only on those (narrow
//! phase). With the `stream` feature the same file can be served from a
//! `RangeReader` without loading all of it.
//!
//! The part worth copying is that the narrow phase runs **in order and stops
//! early**. `raycast_each` visits candidates by nondecreasing box entry `t`,
//! and a box is entered at or before the geometry inside it, so once the
//! stream's entry `t` passes the best exact hit so far, nothing left can beat
//! it. Testing every candidate instead is up to 12.8x slower on a dense mesh
//! (`benches/paired_raycast_prune.rs`), and below roughly five candidates per
//! ray it is the faster shape — the ordered walk keeps a priority queue that
//! the unordered sweep does not.
//!
//! Run: cargo run --example raycast_mesh

use std::ops::ControlFlow;

use packed_spatial_index::{Index3D, Index3DView, Point3D, Ray3D, Triangle3D};

fn main() {
    // Eight stacked 10x10 grids, one per unit of height, so a ray straight down
    // crosses the whole stack and there is something for the early stop to skip.
    let mut tris = Vec::new();
    for layer in 0..8 {
        let z = layer as f64;
        for i in 0..10 {
            for j in 0..10 {
                let (x, y) = (i as f64, j as f64);
                tris.push(Triangle3D::new([x, y, z], [x + 1.0, y, z], [x, y + 1.0, z]));
            }
        }
    }

    // Index over the triangles' bounding boxes (computed for us).
    let index = Index3D::from_triangles(&tris).unwrap();

    // Serialize the index together with the triangles (fixed-width payload).
    let bytes = index.serialize().triangles(&tris).to_bytes().unwrap();
    println!(
        "{} triangles serialized to {} bytes",
        tris.len(),
        bytes.len()
    );

    // Load zero-copy and cast a ray straight down through (4.2, 4.2).
    let view = Index3DView::from_bytes(&bytes).unwrap();
    let ray = Ray3D::new(Point3D::new(4.2, 4.2, 10.0), 0.0, 0.0, -1.0, 100.0);

    // Broad phase: every triangle whose bounding box the ray crosses. Counted
    // here only to have something to compare the exact tests against.
    let candidates = view.raycast(ray).len();

    // Narrow phase, in entry-`t` order, stopping as soon as the order proves
    // nothing left can win.
    let mut tested = 0usize;
    let mut best: Option<(usize, f64)> = None;
    let _: ControlFlow<()> = view.raycast_each(ray, |id, entry_t| {
        if best.is_some_and(|(_, t)| entry_t > t) {
            return ControlFlow::Break(());
        }
        tested += 1;
        let tri = view.triangle::<Triangle3D>(id).expect("triangle payload");
        if let Some(hit) = ray.closest_triangle(&[tri])
            && best.is_none_or(|(_, t)| hit.t < t)
        {
            best = Some((id, hit.t));
        }
        ControlFlow::Continue(())
    });

    println!("broad phase: {candidates} candidate boxes, exact tests run: {tested}");
    match best {
        // dir is length 1 (0,0,-1), so the hit z is origin.z - t.
        Some((id, t)) => println!("hit triangle #{id} at t = {t:.3} (z = {:.2})", 10.0 - t),
        None => println!("ray missed the mesh"),
    }
}
