//! Ray-triangle closest hit over a candidate slice: `f64` (`Triangle3D`) vs
//! compact `f32` (`Triangle3DF32`). This is the exact narrow-phase test a mesh
//! BVH runs on the candidates a `raycast` broad phase returns. The `f32` path
//! uses the SIMD kernel with the `simd` feature; `f64` is scalar.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use packed_spatial_index::{Point3D, Ray3D, Triangle3D, Triangle3DF32};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const N: usize = 4_096; // candidate triangles
const RAYS: usize = 4_096;

fn scene(seed: u64) -> (Vec<Triangle3D>, Vec<Triangle3DF32>, Vec<Ray3D>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut t64 = Vec::with_capacity(N);
    let mut t32 = Vec::with_capacity(N);
    for _ in 0..N {
        let c = [
            rng.random_range(0.0..100.0f64),
            rng.random_range(0.0..100.0),
            rng.random_range(0.0..100.0),
        ];
        let mut v = || {
            [
                c[0] + rng.random_range(-1.0..1.0f64),
                c[1] + rng.random_range(-1.0..1.0),
                c[2] + rng.random_range(-1.0..1.0),
            ]
        };
        let (a, b, cc) = (v(), v(), v());
        t64.push(Triangle3D::new(a, b, cc));
        t32.push(Triangle3DF32::new(
            [a[0] as f32, a[1] as f32, a[2] as f32],
            [b[0] as f32, b[1] as f32, b[2] as f32],
            [cc[0] as f32, cc[1] as f32, cc[2] as f32],
        ));
    }
    let rays = (0..RAYS)
        .map(|_| {
            Ray3D::new(
                Point3D::new(
                    rng.random_range(0.0..100.0),
                    rng.random_range(0.0..100.0),
                    -10.0,
                ),
                rng.random_range(-0.5..0.5),
                rng.random_range(-0.5..0.5),
                1.0,
                1_000.0,
            )
        })
        .collect();
    (t64, t32, rays)
}

fn closest_triangle_benches(c: &mut Criterion) {
    let (t64, t32, rays) = scene(0x7213A);
    let mut group = c.benchmark_group("closest_triangle");
    group.bench_function("f64_Triangle3D", |b| {
        b.iter(|| {
            let mut acc = 0.0f64;
            for ray in &rays {
                if let Some(h) = ray.closest_triangle(&t64) {
                    acc += h.t;
                }
            }
            black_box(acc)
        });
    });
    group.bench_function("f32_Triangle3DF32", |b| {
        b.iter(|| {
            let mut acc = 0.0f64;
            for ray in &rays {
                if let Some(h) = ray.closest_triangle(&t32) {
                    acc += h.t;
                }
            }
            black_box(acc)
        });
    });
    group.finish();
}

criterion_group!(benches, closest_triangle_benches);
criterion_main!(benches);
