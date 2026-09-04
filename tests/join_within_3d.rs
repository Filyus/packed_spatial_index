use std::collections::BTreeSet;
use std::ops::ControlFlow;

use packed_spatial_index::{Box3D, Index3D, Index3DBuilder, Index3DView};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn random_boxes(rng: &mut StdRng, count: usize, extent: f64, max_size: f64) -> Vec<Box3D> {
    (0..count)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..extent);
            let y: f64 = rng.random_range(0.0..extent);
            let z: f64 = rng.random_range(0.0..extent);
            let w: f64 = rng.random_range(0.0..max_size);
            let h: f64 = rng.random_range(0.0..max_size);
            let d: f64 = rng.random_range(0.0..max_size);
            Box3D::new(x, y, z, x + w, y + h, z + d)
        })
        .collect()
}

fn build(boxes: &[Box3D]) -> Index3D {
    let mut builder = Index3DBuilder::new(boxes.len());
    for &b in boxes {
        builder.add(b);
    }
    builder.finish().unwrap()
}

fn naive_epsilon_join(a: &[Box3D], b: &[Box3D], epsilon: f64) -> BTreeSet<(usize, usize)> {
    let mut out = BTreeSet::new();
    for (i, box_a) in a.iter().enumerate() {
        for (j, box_b) in b.iter().enumerate() {
            if box_a.distance_to_box(*box_b) <= epsilon {
                out.insert((i, j));
            }
        }
    }
    out
}

fn naive_self_epsilon_join(boxes: &[Box3D], epsilon: f64) -> BTreeSet<(usize, usize)> {
    let mut out = BTreeSet::new();
    for i in 0..boxes.len() {
        for j in (i + 1)..boxes.len() {
            if boxes[i].distance_to_box(boxes[j]) <= epsilon {
                out.insert((i, j));
            }
        }
    }
    out
}

fn naive_anti_join(a: &[Box3D], b: &[Box3D], epsilon: f64) -> BTreeSet<usize> {
    let paired: BTreeSet<usize> = naive_epsilon_join(a, b, epsilon)
        .into_iter()
        .map(|(i, _)| i)
        .collect();
    (0..a.len()).filter(|i| !paired.contains(i)).collect()
}

/// Components of the epsilon-proximity graph over brute-force pairs, labeled
/// by smallest member id.
fn naive_components(boxes: &[Box3D], epsilon: f64) -> Vec<usize> {
    let mut parent: Vec<usize> = (0..boxes.len()).collect();
    fn find(parent: &[usize], mut x: usize) -> usize {
        while parent[x] != x {
            x = parent[x];
        }
        x
    }
    for i in 0..boxes.len() {
        for j in (i + 1)..boxes.len() {
            if boxes[i].distance_to_box(boxes[j]) <= epsilon {
                let (ra, rb) = (find(&parent, i), find(&parent, j));
                if ra < rb {
                    parent[rb] = ra;
                } else if rb < ra {
                    parent[ra] = rb;
                }
            }
        }
    }
    (0..boxes.len()).map(|x| find(&parent, x)).collect()
}

fn normalized(pairs: Vec<(usize, usize)>) -> BTreeSet<(usize, usize)> {
    let normalized: BTreeSet<_> = pairs.iter().map(|&(i, j)| (i.min(j), i.max(j))).collect();
    assert_eq!(normalized.len(), pairs.len(), "duplicate pairs reported");
    normalized
}

fn sorted_ids(mut ids: Vec<usize>) -> Vec<usize> {
    ids.sort_unstable();
    ids
}

#[test]
fn epsilon_join_matches_naive_pairs_3d() {
    let mut rng = StdRng::seed_from_u64(1212);
    for (n, m, max_size, epsilon) in [
        (0, 7, 4.0, 2.0),
        (1, 1, 4.0, 0.0),
        (37, 5, 8.0, 5.0),
        (400, 300, 3.0, 1.0),
        (400, 300, 3.0, 6.0),
    ] {
        let boxes_a = random_boxes(&mut rng, n, 100.0, max_size);
        let boxes_b = random_boxes(&mut rng, m, 100.0, max_size);
        let a = build(&boxes_a);
        let b = build(&boxes_b);

        let expected = naive_epsilon_join(&boxes_a, &boxes_b, epsilon);
        let actual: BTreeSet<_> = a.join_within(&b, epsilon).into_iter().collect();
        assert_eq!(
            a.join_within(&b, epsilon).len(),
            expected.len(),
            "duplicate pairs reported (n={n} m={m} eps={epsilon})"
        );
        assert_eq!(actual, expected, "n={n} m={m} eps={epsilon}");
    }
}

#[test]
fn epsilon_self_join_matches_naive_pairs_3d() {
    let mut rng = StdRng::seed_from_u64(1313);
    for (n, max_size, epsilon) in [
        (0, 4.0, 2.0),
        (1, 4.0, 2.0),
        (33, 8.0, 4.0),
        (600, 2.5, 1.5),
    ] {
        let boxes = random_boxes(&mut rng, n, 100.0, max_size);
        let index = build(&boxes);

        let expected = naive_self_epsilon_join(&boxes, epsilon);
        assert_eq!(
            normalized(index.self_join_within(epsilon)),
            expected,
            "n={n} eps={epsilon}"
        );
    }
}

#[test]
fn epsilon_zero_equals_overlap_join_3d() {
    let mut rng = StdRng::seed_from_u64(1414);
    let boxes_a = random_boxes(&mut rng, 250, 100.0, 6.0);
    let boxes_b = random_boxes(&mut rng, 200, 100.0, 6.0);
    let a = build(&boxes_a);
    let b = build(&boxes_b);

    let joined: BTreeSet<_> = a.join(&b).into_iter().collect();
    let joined_at_zero: BTreeSet<_> = a.join_within(&b, 0.0).into_iter().collect();
    assert_eq!(joined, joined_at_zero);

    let self_joined = normalized(a.self_join());
    let self_joined_at_zero = normalized(a.self_join_within(0.0));
    assert_eq!(self_joined, self_joined_at_zero);
}

#[test]
fn epsilon_boundary_is_inclusive_3d() {
    // Exactly epsilon apart on x, overlapping spans on y and z.
    let a = build(&[Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)]);
    let b = build(&[Box3D::new(3.0, 0.0, 0.0, 4.0, 1.0, 1.0)]);
    assert_eq!(a.join_within(&b, 2.0), vec![(0, 0)]);
    assert!(a.join_within(&b, 1.999_999).is_empty());
}

#[test]
fn epsilon_invalid_matches_nothing_3d() {
    let mut rng = StdRng::seed_from_u64(1515);
    let boxes = random_boxes(&mut rng, 40, 50.0, 4.0);
    let a = build(&boxes);
    let b = build(&random_boxes(&mut rng, 30, 50.0, 4.0));

    for epsilon in [-1.0, f64::NAN] {
        assert!(a.join_within(&b, epsilon).is_empty());
        assert!(a.self_join_within(epsilon).is_empty());
        assert_eq!(
            a.anti_join_within(&b, epsilon).len(),
            boxes.len(),
            "eps={epsilon}"
        );
        assert_eq!(
            a.self_join_within_components(epsilon),
            (0..boxes.len()).collect::<Vec<_>>(),
            "eps={epsilon}"
        );
    }
}

#[test]
fn epsilon_huge_reports_every_pair_3d() {
    let mut rng = StdRng::seed_from_u64(1616);
    let boxes_a = random_boxes(&mut rng, 12, 50.0, 3.0);
    let boxes_b = random_boxes(&mut rng, 9, 50.0, 3.0);
    let a = build(&boxes_a);
    let b = build(&boxes_b);

    assert_eq!(a.join_within(&b, 1.0e9).len(), 12 * 9);
    assert_eq!(a.self_join_within(1.0e9).len(), 12 * 11 / 2);
}

#[test]
fn epsilon_join_with_supports_early_exit_3d() {
    let mut rng = StdRng::seed_from_u64(1717);
    let boxes = random_boxes(&mut rng, 400, 60.0, 6.0);
    let index = build(&boxes);

    let total = index.self_join_within(5.0).len();
    assert!(total > 10, "test needs a pair-rich input, got {total}");

    let mut seen = 0usize;
    let flow = index.self_join_within_with(5.0, |_, _| {
        seen += 1;
        if seen == 10 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    assert_eq!(flow, ControlFlow::Break(()));
    assert_eq!(seen, 10);
}

#[test]
fn epsilon_components_match_naive_union_find_3d() {
    let mut rng = StdRng::seed_from_u64(1818);
    for (max_size, epsilon) in [(8.0, 2.0), (8.0, 5.0), (20.0, 10.0)] {
        let boxes = random_boxes(&mut rng, 300, 100.0, max_size);
        let index = build(&boxes);
        assert_eq!(
            index.self_join_within_components(epsilon),
            naive_components(&boxes, epsilon),
            "max_size={max_size} eps={epsilon}"
        );
    }
}

#[test]
fn epsilon_chain_is_one_component_3d() {
    let boxes = [
        Box3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        Box3D::new(2.0, 0.0, 0.0, 3.0, 1.0, 1.0),
        Box3D::new(4.0, 0.0, 0.0, 5.0, 1.0, 1.0),
    ];
    let index = build(&boxes);
    assert_eq!(index.self_join_within_components(1.0), vec![0, 0, 0]);
    assert!(index.anti_join_within(&index, 1.0).is_empty());
}

#[test]
fn epsilon_anti_join_matches_naive_3d() {
    let mut rng = StdRng::seed_from_u64(1919);
    for epsilon in [0.5, 3.0, 10.0] {
        let boxes_a = random_boxes(&mut rng, 250, 100.0, 5.0);
        let boxes_b = random_boxes(&mut rng, 180, 100.0, 5.0);
        let a = build(&boxes_a);
        let b = build(&boxes_b);
        assert_eq!(
            sorted_ids(a.anti_join_within(&b, epsilon)),
            naive_anti_join(&boxes_a, &boxes_b, epsilon)
                .into_iter()
                .collect::<Vec<_>>(),
            "eps={epsilon}"
        );
    }
}

#[test]
fn view_epsilon_family_matches_owned_3d() {
    let mut rng = StdRng::seed_from_u64(2020);
    let boxes_a = random_boxes(&mut rng, 250, 100.0, 6.0);
    let boxes_b = random_boxes(&mut rng, 180, 100.0, 6.0);
    let a = build(&boxes_a);
    let b = build(&boxes_b);
    let bytes_a = a.to_bytes();
    let bytes_b = b.to_bytes();
    let view_a = Index3DView::from_bytes(&bytes_a).unwrap();
    let view_b = Index3DView::from_bytes(&bytes_b).unwrap();

    for epsilon in [0.0, 2.5] {
        let owned: BTreeSet<_> = a.join_within(&b, epsilon).into_iter().collect();
        let viewed: BTreeSet<_> = view_a.join_within(&view_b, epsilon).into_iter().collect();
        assert_eq!(owned, viewed, "eps={epsilon}");
        assert_eq!(
            normalized(view_a.self_join_within(epsilon)),
            normalized(a.self_join_within(epsilon)),
            "eps={epsilon}"
        );
        assert_eq!(
            sorted_ids(view_a.anti_join_within(&view_b, epsilon)),
            sorted_ids(a.anti_join_within(&b, epsilon)),
            "eps={epsilon}"
        );
    }
    assert_eq!(
        view_a.self_join_within_components(3.0),
        a.self_join_within_components(3.0)
    );
}

#[cfg(feature = "simd")]
mod simd {
    use super::*;
    use packed_spatial_index::SimdIndex3DView;

    fn build_simd(boxes: &[Box3D]) -> packed_spatial_index::SimdIndex3D {
        let mut builder = Index3DBuilder::new(boxes.len());
        for &b in boxes {
            builder.add(b);
        }
        builder.finish_simd().unwrap()
    }

    #[test]
    fn simd_epsilon_family_matches_naive_3d() {
        let mut rng = StdRng::seed_from_u64(2121);
        for epsilon in [0.0, 1.5, 5.0] {
            let boxes_a = random_boxes(&mut rng, 450, 100.0, 4.0);
            let boxes_b = random_boxes(&mut rng, 350, 100.0, 4.0);
            let a = build_simd(&boxes_a);
            let b = build_simd(&boxes_b);

            let expected = naive_epsilon_join(&boxes_a, &boxes_b, epsilon);
            let actual: BTreeSet<_> = a.join_within(&b, epsilon).into_iter().collect();
            assert_eq!(actual, expected, "eps={epsilon}");

            let expected_self = naive_self_epsilon_join(&boxes_a, epsilon);
            assert_eq!(normalized(a.self_join_within(epsilon)), expected_self);

            assert_eq!(
                sorted_ids(a.anti_join_within(&b, epsilon)),
                naive_anti_join(&boxes_a, &boxes_b, epsilon)
                    .into_iter()
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                a.self_join_within_components(epsilon),
                naive_components(&boxes_a, epsilon),
                "eps={epsilon}"
            );
        }
    }

    #[test]
    fn simd_view_epsilon_family_matches_owned_3d() {
        let mut rng = StdRng::seed_from_u64(2222);
        let boxes_a = random_boxes(&mut rng, 200, 100.0, 5.0);
        let boxes_b = random_boxes(&mut rng, 240, 100.0, 5.0);
        let a = build_simd(&boxes_a);
        let b = build_simd(&boxes_b);
        let bytes_a = a.to_bytes();
        let bytes_b = b.to_bytes();
        let view_a = SimdIndex3DView::from_bytes(&bytes_a).unwrap();
        let view_b = SimdIndex3DView::from_bytes(&bytes_b).unwrap();

        for epsilon in [0.0, 3.0] {
            let owned: BTreeSet<_> = a.join_within(&b, epsilon).into_iter().collect();
            let viewed: BTreeSet<_> = view_a.join_within(&view_b, epsilon).into_iter().collect();
            assert_eq!(owned, viewed, "eps={epsilon}");
            assert_eq!(
                normalized(view_a.self_join_within(epsilon)),
                normalized(a.self_join_within(epsilon))
            );
            assert_eq!(
                sorted_ids(view_a.anti_join_within(&view_b, epsilon)),
                sorted_ids(a.anti_join_within(&b, epsilon))
            );
        }
        assert_eq!(
            view_a.self_join_within_components(4.0),
            a.self_join_within_components(4.0)
        );
    }
}
