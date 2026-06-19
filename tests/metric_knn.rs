//! Custom-metric kNN (`neighbors_metric` / `visit_neighbors_metric`) on the owned
//! f64 indexes and their zero-copy views must return exactly the brute-force
//! k-nearest under the same metric, and the byte views must match the owned
//! indexes. Also checks the `haversine_distance_2d` helper against a known value.

use packed_spatial_index::{
    Box2D, Box3D, EARTH_RADIUS_M, Index2DBuilder, Index2DView, Index3DBuilder, Index3DView,
    Point2D, haversine_distance_2d,
};
use std::ops::ControlFlow;

/// A degenerate (point) box at `(x, y)`.
fn bp(x: f64, y: f64) -> Box2D {
    Box2D::from_point(Point2D::new(x, y))
}

fn boxes2(n: usize, seed: u64) -> Vec<Box2D> {
    (0..n)
        .map(|i| {
            let h = (i.wrapping_mul(2654435761) ^ seed as usize) as u32;
            let x = (h % 1009) as f64;
            let y = ((h >> 8) % 1013) as f64;
            let w = 0.5 + ((h >> 3) % 7) as f64;
            Box2D::new(x, y, x + w, y + 0.5 + ((h >> 5) % 6) as f64)
        })
        .collect()
}

fn boxes3(n: usize, seed: u64) -> Vec<Box3D> {
    (0..n)
        .map(|i| {
            let h = (i.wrapping_mul(2654435761) ^ seed as usize) as u32;
            let x = (h % 1009) as f64;
            let y = ((h >> 8) % 1013) as f64;
            let z = ((h >> 16) % 1019) as f64;
            Box3D::new(x, y, z, x + 1.5, y + 2.0, z + 1.0)
        })
        .collect()
}

/// Euclidean distance from point `p` to the closest point of box `b`.
fn dist2(p: (f64, f64), b: Box2D) -> f64 {
    let dx = (b.min_x - p.0).max(0.0).max(p.0 - b.max_x);
    let dy = (b.min_y - p.1).max(0.0).max(p.1 - b.max_y);
    (dx * dx + dy * dy).sqrt()
}

/// Manhattan distance from point `p` to the closest point of box `b` (3D).
fn manhattan3(p: (f64, f64, f64), b: Box3D) -> f64 {
    let dx = (b.min_x - p.0).max(0.0).max(p.0 - b.max_x);
    let dy = (b.min_y - p.1).max(0.0).max(p.1 - b.max_y);
    let dz = (b.min_z - p.2).max(0.0).max(p.2 - b.max_z);
    dx + dy + dz
}

/// Brute-force k indices nearest to the query by `metric`, ties broken by index.
fn brute<T: Copy>(boxes: &[T], k: usize, metric: impl Fn(T) -> f64) -> Vec<usize> {
    let mut d: Vec<(f64, usize)> = boxes
        .iter()
        .enumerate()
        .map(|(i, &b)| (metric(b), i))
        .collect();
    d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    d.into_iter().take(k).map(|(_, i)| i).collect()
}

fn sorted(mut v: Vec<usize>) -> Vec<usize> {
    v.sort_unstable();
    v
}

#[test]
fn metric_matches_bruteforce_2d() {
    for &n in &[1usize, 5, 17, 64, 1000] {
        for &ns in &[4usize, 16] {
            let bs = boxes2(n, 7);
            let mut b = Index2DBuilder::new(n).node_size(ns);
            bs.iter().for_each(|&x| b.add(x));
            let index = b.finish().unwrap();

            for q in [(10.0, 10.0), (500.0, 500.0), (0.0, 0.0), (1008.0, 1012.0)] {
                for &k in &[1usize, 3, 10] {
                    let got = index.neighbors_metric(|bx| dist2(q, bx), k, f64::INFINITY);
                    let exp = brute(&bs, k.min(n), |bx| dist2(q, bx));
                    // Distinct distances on this data => exact set match.
                    assert_eq!(
                        sorted(got.clone()),
                        sorted(exp),
                        "n={n} ns={ns} q={q:?} k={k}"
                    );
                    // Returned in nondecreasing metric distance.
                    let ds: Vec<f64> = got.iter().map(|&i| dist2(q, bs[i])).collect();
                    assert!(ds.windows(2).all(|w| w[0] <= w[1]), "not sorted: {ds:?}");
                    // visit yields the same set.
                    let mut vis = Vec::new();
                    let _ = index.visit_neighbors_metric::<(), _, _>(
                        |bx| dist2(q, bx),
                        f64::INFINITY,
                        |i, _| {
                            vis.push(i);
                            if vis.len() == k {
                                ControlFlow::Break(())
                            } else {
                                ControlFlow::Continue(())
                            }
                        },
                    );
                    assert_eq!(sorted(vis), sorted(got), "visit != collect");
                }
            }
        }
    }
}

#[test]
fn metric_matches_bruteforce_3d() {
    for &n in &[1usize, 17, 1000] {
        let bs = boxes3(n, 9);
        let mut b = Index3DBuilder::new(n).node_size(8);
        bs.iter().for_each(|&x| b.add(x));
        let index = b.finish().unwrap();

        for q in [(10.0, 10.0, 10.0), (500.0, 500.0, 500.0)] {
            for &k in &[1usize, 5] {
                let got = index.neighbors_metric(|bx| manhattan3(q, bx), k, f64::INFINITY);
                let exp = brute(&bs, k.min(n), |bx| manhattan3(q, bx));
                assert_eq!(sorted(got), sorted(exp), "3d n={n} q={q:?} k={k}");
            }
        }
    }
}

#[test]
fn view_matches_owned() {
    let bs = boxes2(2000, 3);
    let mut b = Index2DBuilder::new(bs.len());
    bs.iter().for_each(|&x| b.add(x));
    let owned = b.finish().unwrap();
    let bytes = owned.to_bytes();
    let view = Index2DView::from_bytes(&bytes).unwrap();

    let bs3 = boxes3(2000, 4);
    let mut b3 = Index3DBuilder::new(bs3.len());
    bs3.iter().for_each(|&x| b3.add(x));
    let owned3 = b3.finish().unwrap();
    let bytes3 = owned3.to_bytes();
    let view3 = Index3DView::from_bytes(&bytes3).unwrap();

    for q in [(123.0, 456.0), (12.0, 900.0)] {
        let o = owned.neighbors_metric(|bx| dist2(q, bx), 7, f64::INFINITY);
        let v = view.neighbors_metric(|bx| dist2(q, bx), 7, f64::INFINITY);
        assert_eq!(o, v, "2d view != owned at {q:?}");
    }
    let q3 = (100.0, 200.0, 300.0);
    assert_eq!(
        owned3.neighbors_metric(|bx| manhattan3(q3, bx), 5, f64::INFINITY),
        view3.neighbors_metric(|bx| manhattan3(q3, bx), 5, f64::INFINITY),
    );
}

#[test]
fn max_distance_cutoff() {
    let bs = boxes2(500, 11);
    let mut b = Index2DBuilder::new(bs.len());
    bs.iter().for_each(|&x| b.add(x));
    let index = b.finish().unwrap();
    let q = (500.0, 500.0);
    let cutoff = 50.0;
    let got = index.neighbors_metric(|bx| dist2(q, bx), 1000, cutoff);
    // Every returned item is within the cutoff, and none within is omitted.
    assert!(got.iter().all(|&i| dist2(q, bs[i]) <= cutoff));
    let expected = bs.iter().filter(|&&bx| dist2(q, bx) <= cutoff).count();
    assert_eq!(got.len(), expected);
}

#[test]
fn empty_and_degenerate() {
    let index = Index2DBuilder::new(0).finish().unwrap();
    assert!(
        index
            .neighbors_metric(|bx| dist2((0.0, 0.0), bx), 5, f64::INFINITY)
            .is_empty()
    );

    let mut b = Index2DBuilder::new(3);
    for &x in &boxes2(3, 1) {
        b.add(x);
    }
    let index = b.finish().unwrap();
    assert!(
        index
            .neighbors_metric(|bx| dist2((0.0, 0.0), bx), 0, f64::INFINITY)
            .is_empty()
    );
    // NaN / negative cutoff => nothing.
    assert!(
        index
            .neighbors_metric(|bx| dist2((0.0, 0.0), bx), 5, f64::NAN)
            .is_empty()
    );
    assert!(
        index
            .neighbors_metric(|bx| dist2((0.0, 0.0), bx), 5, -1.0)
            .is_empty()
    );
}

#[test]
fn haversine_known_distance() {
    // Berlin (13.405, 52.52) to Paris (2.3522, 48.8566): ~878 km great-circle.
    let paris = bp(2.3522, 48.8566);
    let d = haversine_distance_2d((13.405, 52.52), paris, EARTH_RADIUS_M);
    assert!(
        (d - 878_000.0).abs() < 15_000.0,
        "haversine Berlin-Paris = {d}"
    );

    // Distance to a box you're inside is 0.
    let inside = Box2D::new(0.0, 0.0, 10.0, 10.0);
    assert_eq!(
        haversine_distance_2d((5.0, 5.0), inside, EARTH_RADIUS_M),
        0.0
    );
}

#[test]
fn haversine_knn_picks_nearest_city() {
    // Query near Berlin; nearest of a few European capitals must be Berlin.
    let cities = [
        bp(2.3522, 48.8566),  // 0 Paris
        bp(13.405, 52.52),    // 1 Berlin
        bp(-0.1276, 51.5072), // 2 London
        bp(12.4964, 41.9028), // 3 Rome
    ];
    let mut b = Index2DBuilder::new(cities.len());
    cities.iter().for_each(|&c| b.add(c));
    let index = b.finish().unwrap();

    let q = (13.0, 52.4); // just SW of Berlin
    let got = index.neighbors_metric(
        |bx| haversine_distance_2d(q, bx, EARTH_RADIUS_M),
        2,
        f64::INFINITY,
    );
    let exp = brute(&cities, 2, |bx| {
        haversine_distance_2d(q, bx, EARTH_RADIUS_M)
    });
    assert_eq!(got, exp);
    assert_eq!(got[0], 1, "nearest must be Berlin");
}
