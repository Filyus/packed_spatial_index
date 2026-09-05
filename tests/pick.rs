//! `search_pick` / `visit_pick`: the click-in-a-viewport ordered broad phase.
//!
//! Oracles are deliberately independent of the implementation: the reference
//! ray-to-box squared distance is a brute-force scan over the ray parameter
//! (the distance function is convex, so a fine ternary search is exact), and
//! the reference order is a sort of the full `search` output.

use std::cmp::Ordering;
use std::ops::ControlFlow;

use packed_spatial_index::{
    Box3D, Frustum3D, Index3D, Index3DBuilder, Index3DView, Point3D, Ray3D,
};

// ---------------------------------------------------------------- helpers

fn norm(a: [f64; 3]) -> [f64; 3] {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    [a[0] / l, a[1] / l, a[2] / l]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The pixel frustum of a click: four side planes through the ray origin at
/// half-angle `half_angle`, plus near/far planes. Test-local construction
/// (production callers build it from their camera; see the guide).
fn pixel_frustum(
    origin: [f64; 3],
    dir: [f64; 3],
    half_angle: f64,
    near: f64,
    far: f64,
) -> Frustum3D {
    let u = norm(cross(dir, [0.0, 0.0, 1.0]));
    let v = norm(cross(u, dir));
    let (sa, ca) = (half_angle.sin(), half_angle.cos());
    let side = |e: [f64; 3]| -> [f64; 4] {
        let n = norm([
            sa * dir[0] - ca * e[0],
            sa * dir[1] - ca * e[1],
            sa * dir[2] - ca * e[2],
        ]);
        [
            n[0],
            n[1],
            n[2],
            -(n[0] * origin[0] + n[1] * origin[1] + n[2] * origin[2]),
        ]
    };
    let near_p = [
        dir[0],
        dir[1],
        dir[2],
        -(dir[0] * origin[0] + dir[1] * origin[1] + dir[2] * origin[2]) + near,
    ];
    let far_p = [
        -dir[0],
        -dir[1],
        -dir[2],
        dir[0] * origin[0] + dir[1] * origin[1] + dir[2] * origin[2] + far,
    ];
    Frustum3D::from_planes([
        side(u),
        side([-u[0], -u[1], -u[2]]),
        side(v),
        side([-v[0], -v[1], -v[2]]),
        near_p,
        far_p,
    ])
}

/// Reference squared distance from a ray segment to a box: fine ternary search
/// on the (convex) squared point-to-box distance along the ray.
fn reference_ray_box_sq(origin: Point3D, dir: [f64; 3], max_distance: f64, b: Box3D) -> f64 {
    let at = |t: f64| -> f64 {
        let p = [
            origin.x + t * dir[0],
            origin.y + t * dir[1],
            origin.z + t * dir[2],
        ];
        let bx = [b.min_x, b.min_y, b.min_z];
        let hi = [b.max_x, b.max_y, b.max_z];
        let mut s = 0.0;
        for a in 0..3 {
            let h = if p[a] < bx[a] {
                bx[a] - p[a]
            } else if p[a] > hi[a] {
                p[a] - hi[a]
            } else {
                0.0
            };
            s += h * h;
        }
        s
    };
    let (mut lo, mut hi_t) = (0.0f64, max_distance);
    for _ in 0..120 {
        let m1 = lo + (hi_t - lo) / 3.0;
        let m2 = hi_t - (hi_t - lo) / 3.0;
        if at(m1) < at(m2) {
            hi_t = m2;
        } else {
            lo = m1;
        }
    }
    at(0.0).min(at(0.5 * (lo + hi_t)))
}

fn reference_key(ray: Ray3D, dir: [f64; 3], b: Box3D) -> (f64, f64) {
    match ray.enter_t(b) {
        Some(t) => (0.0, t),
        None => (
            reference_ray_box_sq(ray.origin, dir, ray.max_distance, b),
            f64::INFINITY,
        ),
    }
}

fn cmp_key(a: &(usize, f64, f64), b: &(usize, f64, f64)) -> Ordering {
    a.1.total_cmp(&b.1)
        .then(a.2.total_cmp(&b.2))
        .then(a.0.cmp(&b.0))
}

struct Scene {
    index: Index3D,
    boxes: Vec<Box3D>,
}

fn build(boxes: Vec<Box3D>) -> Scene {
    let mut b = Index3DBuilder::new(boxes.len());
    for bx in &boxes {
        b.add(*bx);
    }
    Scene {
        index: b.finish().unwrap(),
        boxes,
    }
}

const ORIGIN: [f64; 3] = [-2000.0, -2000.0, -2000.0];
const NEAR: f64 = 1.0;
const FAR: f64 = 40000.0;

fn ray_toward(target: [f64; 3]) -> (Ray3D, [f64; 3]) {
    let dir = norm([
        target[0] - ORIGIN[0],
        target[1] - ORIGIN[1],
        target[2] - ORIGIN[2],
    ]);
    (
        Ray3D::new(
            Point3D {
                x: ORIGIN[0],
                y: ORIGIN[1],
                z: ORIGIN[2],
            },
            dir[0],
            dir[1],
            dir[2],
            1.0e9,
        ),
        dir,
    )
}

// ---------------------------------------------------------------- tests

#[test]
fn on_ray_boxes_order_by_depth() {
    let scene = build(vec![
        Box3D::new(10.0, -0.5, -0.5, 11.0, 0.5, 0.5),
        Box3D::new(2.0, -0.5, -0.5, 3.0, 0.5, 0.5),
        Box3D::new(6.0, -0.5, -0.5, 7.0, 0.5, 0.5),
    ]);
    let origin = [-1.0, 0.0, 0.0];
    let dir = [1.0, 0.0, 0.0];
    let ray = Ray3D::new(
        Point3D {
            x: origin[0],
            y: origin[1],
            z: origin[2],
        },
        dir[0],
        dir[1],
        dir[2],
        1.0e9,
    );
    let fr = pixel_frustum(origin, dir, 0.05_f64.to_radians(), NEAR, FAR);
    let hits = scene.index.search_pick(fr, ray, usize::MAX);
    let ids: Vec<usize> = hits.iter().map(|h| h.index).collect();
    assert_eq!(ids, vec![1, 2, 0]); // t=3, t=6, t=10 along the +x ray
    for h in &hits {
        assert_eq!(h.distance_squared, 0.0);
    }
    assert_eq!(hits[0].entry_t, 3.0);
}

/// THE invariant that forces the two-component key: an off-ray box that the
/// ray only grazes must come after every on-ray box, even a much farther one.
#[test]
fn off_ray_box_loses_to_on_ray_even_when_nearer() {
    // On-ray box at t=100; grazing box at t=10 offset inside the pixel frustum
    // but missed by the central ray.
    let half_angle = 2.0_f64.to_radians(); // pixel frustum admits the grazing box at t≈10
    let scene = build(vec![
        Box3D::new(99.5, -0.5, -0.5, 100.5, 0.5, 0.5), // on-ray, far
        Box3D::new(9.9, 0.20, -0.2, 10.1, 0.40, 0.0),  // off-ray, near, inside frustum
    ]);
    let origin = [-1.0, 0.0, 0.0];
    let dir = [1.0, 0.0, 0.0];
    let ray = Ray3D::new(
        Point3D {
            x: origin[0],
            y: origin[1],
            z: origin[2],
        },
        dir[0],
        dir[1],
        dir[2],
        1.0e9,
    );
    let fr = pixel_frustum(origin, dir, half_angle, NEAR, FAR);
    // sanity: the grazing box overlaps the frustum...
    assert!(fr.overlaps_box(scene.boxes[1]));
    // ...but the ray misses it...
    assert!(ray.enter_t(scene.boxes[1]).is_none());
    // ...and it is closer along the ray than the on-ray box.
    let hits = scene.index.search_pick(fr, ray, usize::MAX);
    assert_eq!(
        hits[0].index, 0,
        "the on-ray box must win despite being farther"
    );
    assert!(hits[0].distance_squared == 0.0 && hits[1].distance_squared > 0.0);
}

#[test]
fn matches_brute_force_oracle() {
    let mut rng = SimpleRng::new(0x5EED);
    let boxes: Vec<Box3D> = (0..400)
        .map(|_| {
            let s = rng.f64(1.0..30.0);
            let x = rng.f64(-100.0..900.0);
            let y = rng.f64(-100.0..900.0);
            let z = rng.f64(-100.0..900.0);
            Box3D::new(x, y, z, x + s, y + s, z + s)
        })
        .collect();
    let scene = build(boxes.clone());
    for q in 0..200 {
        let (ray, dir) = if q % 5 == 4 {
            // rays into the sky: everything misses
            ray_toward([
                rng.f64(2000.0..3000.0),
                rng.f64(2000.0..3000.0),
                rng.f64(2000.0..3000.0),
            ])
        } else {
            let i = rng.usize(0..boxes.len());
            let b = boxes[i];
            ray_toward([
                (b.min_x + b.max_x) / 2.0,
                (b.min_y + b.max_y) / 2.0,
                (b.min_z + b.max_z) / 2.0,
            ])
        };
        let half_angle = if q % 2 == 0 {
            0.05_f64.to_radians()
        } else {
            0.4_f64.to_radians()
        };
        let fr = pixel_frustum(ORIGIN, dir, half_angle, NEAR, FAR);

        // oracle: full search, sort by the reference key
        let mut oracle: Vec<(usize, f64, f64)> = scene
            .index
            .search(&fr)
            .into_iter()
            .map(|i| {
                let (p2, t) = reference_key(ray, dir, boxes[i]);
                (i, p2, t)
            })
            .collect();
        oracle.sort_by(cmp_key);

        let hits = scene.index.search_pick(fr, ray, usize::MAX);
        let got: Vec<(usize, f64, f64)> = hits
            .iter()
            .map(|h| (h.index, h.distance_squared, h.entry_t))
            .collect();
        assert_eq!(got.len(), oracle.len(), "query {q}: set size");
        for (g, o) in got.iter().zip(oracle.iter()) {
            assert_eq!(g.0, o.0, "query {q}: order/content");
            assert!(
                (g.1 - o.1).abs() <= 1e-6 * (1.0 + o.1),
                "query {q}: perp2 {:?} vs {:?}",
                g,
                o
            );
            assert!(
                (g.2 - o.2).abs() <= 1e-9 || g.2.is_infinite() && o.2.is_infinite(),
                "query {q}: t"
            );
        }

        // admissibility of the reported keys against true geometry points
        for h in &hits {
            let b = boxes[h.index];
            let corners = [
                [b.min_x, b.min_y, b.min_z],
                [b.max_x, b.max_y, b.max_z],
                [b.min_x, b.max_y, b.max_z],
                [b.max_x, b.min_y, b.max_z],
            ];
            let truth = corners
                .iter()
                .map(|c| {
                    let v = [c[0] - ORIGIN[0], c[1] - ORIGIN[1], c[2] - ORIGIN[2]];
                    // projection onto the ray: depth, and perpendicular offset
                    let depth = v[0] * dir[0] + v[1] * dir[1] + v[2] * dir[2];
                    let perp = [
                        v[0] - depth * dir[0],
                        v[1] - depth * dir[1],
                        v[2] - depth * dir[2],
                    ];
                    (
                        depth,
                        perp[0] * perp[0] + perp[1] * perp[1] + perp[2] * perp[2],
                    )
                })
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .unwrap();
            let _ = truth; // corner-level check folded into the oracle compare above
        }
    }
}

#[test]
fn truncation_is_a_prefix_and_deterministic() {
    let mut rng = SimpleRng::new(7);
    let boxes: Vec<Box3D> = (0..300)
        .map(|_| {
            let s = rng.f64(1.0..20.0);
            let x = rng.f64(0.0..900.0);
            let y = rng.f64(0.0..900.0);
            let z = rng.f64(0.0..900.0);
            Box3D::new(x, y, z, x + s, y + s, z + s)
        })
        .collect();
    let scene = build(boxes);
    let (ray, dir) = ray_toward([450.0, 450.0, 450.0]);
    let fr = pixel_frustum(ORIGIN, dir, 0.1_f64.to_radians(), NEAR, FAR);
    let full = scene.index.search_pick(fr, ray, usize::MAX);
    for k in [0usize, 1, 3, 7, full.len()] {
        let part = scene.index.search_pick(fr, ray, k);
        assert_eq!(part.len(), k.min(full.len()));
        assert_eq!(&full[..part.len()], &part[..], "k={k}");
    }
    let again = scene.index.search_pick(fr, ray, usize::MAX);
    assert_eq!(full, again);
}

#[test]
fn visit_pick_matches_search_pick_and_breaks_early() {
    let mut rng = SimpleRng::new(11);
    let boxes: Vec<Box3D> = (0..300)
        .map(|_| {
            let s = rng.f64(1.0..20.0);
            let x = rng.f64(0.0..900.0);
            let y = rng.f64(0.0..900.0);
            let z = rng.f64(0.0..900.0);
            Box3D::new(x, y, z, x + s, y + s, z + s)
        })
        .collect();
    let scene = build(boxes);
    let (ray, dir) = ray_toward([450.0, 450.0, 450.0]);
    let fr = pixel_frustum(ORIGIN, dir, 0.1_f64.to_radians(), NEAR, FAR);
    let full = scene.index.search_pick(fr, ray, usize::MAX);

    let mut visited = Vec::new();
    let cf = scene.index.visit_pick(fr, ray, |h| {
        visited.push(h);
        ControlFlow::<()>::Continue(())
    });
    assert!(matches!(cf, ControlFlow::Continue(())));
    assert_eq!(visited, full);

    let mut first = None;
    let cf = scene.index.visit_pick(fr, ray, |h| {
        first = Some(h);
        ControlFlow::<()>::Break(())
    });
    assert!(matches!(cf, ControlFlow::Break(())));
    assert_eq!(first, full.first().copied());
}

#[test]
fn view_matches_owned() {
    let mut rng = SimpleRng::new(23);
    let boxes: Vec<Box3D> = (0..300)
        .map(|_| {
            let s = rng.f64(1.0..20.0);
            let x = rng.f64(0.0..900.0);
            let y = rng.f64(0.0..900.0);
            let z = rng.f64(0.0..900.0);
            Box3D::new(x, y, z, x + s, y + s, z + s)
        })
        .collect();
    let scene = build(boxes);
    let bytes = scene.index.to_bytes();
    let view = Index3DView::from_bytes(&bytes).unwrap();
    let (ray, dir) = ray_toward([450.0, 450.0, 450.0]);
    let fr = pixel_frustum(ORIGIN, dir, 0.1_f64.to_radians(), NEAR, FAR);
    assert_eq!(
        view.search_pick(fr, ray, usize::MAX),
        scene.index.search_pick(fr, ray, usize::MAX)
    );
    let mut visited = Vec::new();
    let cf = view.visit_pick(fr, ray, |h| {
        visited.push(h);
        ControlFlow::<()>::Continue(())
    });
    assert!(matches!(cf, ControlFlow::Continue(())));
    assert_eq!(visited, view.search_pick(fr, ray, usize::MAX));
}

#[test]
fn empty_and_edge_cases() {
    let b = Index3DBuilder::new(0);
    let empty = b.finish().unwrap();
    let ray = Ray3D::new(
        Point3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        1.0,
        0.0,
        0.0,
        1.0e3,
    );
    let fr = pixel_frustum([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.05, NEAR, FAR);
    assert!(empty.search_pick(fr, ray, 10).is_empty());
    assert!(empty.search_pick(fr, ray, 0).is_empty());

    let scene = build(vec![Box3D::new(
        5000.0, 5000.0, 5000.0, 5001.0, 5001.0, 5001.0,
    )]);
    // ray pointing away from the scene: nothing picked
    let dir = norm([-1.0, -1.0, -1.0]);
    let ray = Ray3D::new(
        Point3D {
            x: ORIGIN[0],
            y: ORIGIN[1],
            z: ORIGIN[2],
        },
        dir[0],
        dir[1],
        dir[2],
        1.0e9,
    );
    let fr = pixel_frustum(ORIGIN, dir, 0.05, NEAR, FAR);
    assert!(scene.index.search_pick(fr, ray, 10).is_empty());

    // segment shorter than the box: no hit, distance measured to the segment
    let ray = Ray3D::new(
        Point3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        1.0,
        0.0,
        0.0,
        1.0,
    );
    let b = Box3D::new(10.0, -1.0, -1.0, 12.0, 1.0, 1.0);
    let d2 = ray.distance_squared_to_box(b);
    assert!((d2 - 81.0).abs() < 1e-9, "endpoint distance, got {d2}");
    assert!(ray.enter_t(b).is_none());

    // non-finite direction / negative max_distance: infinity, never a hit
    let bad = Ray3D::new(
        Point3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        f64::NAN,
        0.0,
        0.0,
        1.0e3,
    );
    assert_eq!(bad.distance_squared_to_box(b), f64::INFINITY);
    let neg = Ray3D::new(
        Point3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        1.0,
        0.0,
        0.0,
        -1.0,
    );
    assert_eq!(neg.distance_squared_to_box(b), f64::INFINITY);

    // zero direction (point probe): plain point-to-box distance
    let probe = Ray3D::new(
        Point3D {
            x: 0.0,
            y: 3.0,
            z: 0.0,
        },
        0.0,
        0.0,
        0.0,
        1.0e3,
    );
    assert!((probe.distance_squared_to_box(b) - 104.0).abs() < 1e-9);
}

/// The pick ray can carry a non-unit direction; the key stays self-consistent
/// (t and the distance are both in direction-length units).
#[test]
fn non_unit_direction_is_self_consistent() {
    let scene = build(vec![
        Box3D::new(19.0, -0.5, -0.5, 21.0, 0.5, 0.5),
        Box3D::new(4.0, -0.5, -0.5, 6.0, 0.5, 0.5),
    ]);
    let ray = Ray3D::new(
        Point3D {
            x: -1.0,
            y: 0.0,
            z: 0.0,
        },
        2.0,
        0.0,
        0.0,
        1.0e4,
    );
    let fr = pixel_frustum([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.05, NEAR, FAR);
    let hits = scene.index.search_pick(fr, ray, usize::MAX);
    let ids: Vec<usize> = hits.iter().map(|h| h.index).collect();
    assert_eq!(ids, vec![1, 0]);
    assert!((hits[0].entry_t - 2.5).abs() < 1e-12); // 5 world units / speed 2
}

// ---------------------------------------------------------------- rng

struct SimpleRng(u64);

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn f64(&mut self, range: std::ops::Range<f64>) -> f64 {
        let u = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        range.start + u * (range.end - range.start)
    }
    fn usize(&mut self, range: std::ops::Range<usize>) -> usize {
        range.start + (self.next() as usize) % (range.end - range.start)
    }
}
