//! Streamable bounding-volume hierarchy over a triangle mesh.
//!
//! Build an index over each triangle's bounding box, attach the triangles as a
//! fixed-width payload (no offset table), then serialize. The same bytes load
//! back zero-copy: a ray query narrows to candidate triangles through the index
//! (broad phase), and an exact ray-triangle test runs only on those (narrow
//! phase). With the `stream` feature the same file can be served from a
//! `RangeReader` without loading all of it.
//!
//! Run: cargo run --example raycast_mesh

use packed_spatial_index::{Index3D, Index3DView, Point3D, Ray3D, Triangle3D};

fn main() {
    // A 10x10 grid of triangles in the z = 0 plane, with a few raised to z = 3.
    let mut tris = Vec::new();
    for i in 0..10 {
        for j in 0..10 {
            let (x, y) = (i as f64, j as f64);
            let z = if (i + j) % 7 == 0 { 3.0 } else { 0.0 };
            tris.push(Triangle3D::new([x, y, z], [x + 1.0, y, z], [x, y + 1.0, z]));
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

    // Load zero-copy and cast a ray straight down through (4.5, 4.5).
    let view = Index3DView::from_bytes(&bytes).unwrap();
    let ray = Ray3D::new(Point3D::new(4.5, 4.5, 10.0), 0.0, 0.0, -1.0, 100.0);

    // Broad phase: the index returns triangles whose bounding box the ray crosses.
    let candidate_ids = index.raycast(ray);
    println!("broad phase: {} candidate triangles", candidate_ids.len());

    // Narrow phase: the exact ray-triangle test, only on the candidates.
    let candidates: Vec<Triangle3D> = candidate_ids
        .iter()
        .map(|&id| view.triangle(id).expect("triangle payload"))
        .collect();
    match ray.closest_triangle(&candidates) {
        Some(hit) => {
            let id = candidate_ids[hit.index];
            // dir is length 1 (0,0,-1), so the hit z is origin.z - t.
            println!(
                "hit triangle #{id} at t = {:.3} (z = {:.2})",
                hit.t,
                10.0 - hit.t
            );
        }
        None => println!("ray missed the mesh"),
    }
}
