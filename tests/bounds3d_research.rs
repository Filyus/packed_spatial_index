#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::hint::black_box;
use std::time::{Duration, Instant};

use packed_spatial_index::experimental::magic_bits as hilbert2d;
use packed_spatial_index::{Bounds2D, Index2D, Index2DBuilder, Point2D};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const MORTON_BITS_PER_AXIS: u32 = 21;
const MORTON_AXIS_MAX: u32 = (1 << MORTON_BITS_PER_AXIS) - 1;
const HILBERT_BITS_PER_AXIS: u32 = 16;
const HILBERT_AXIS_MAX: u32 = (1 << HILBERT_BITS_PER_AXIS) - 1;
const HILBERT3_LUT: [u8; 192] = build_hilbert3_lut();
const ENCODE_ITEMS: usize = 262_144;
const RESEARCH_ITEMS: usize = 8_192;
const RESEARCH_QUERIES: usize = 128;

#[derive(Clone, Copy, Debug)]
struct Bounds3D {
    min_x: f64,
    min_y: f64,
    min_z: f64,
    max_x: f64,
    max_y: f64,
    max_z: f64,
}

impl Bounds3D {
    const fn new(min_x: f64, min_y: f64, min_z: f64, max_x: f64, max_y: f64, max_z: f64) -> Self {
        Self {
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
        }
    }

    fn overlaps(self, other: Self) -> bool {
        (self.min_x <= other.max_x)
            & (self.max_x >= other.min_x)
            & (self.min_y <= other.max_y)
            & (self.max_y >= other.min_y)
            & (self.min_z <= other.max_z)
            & (self.max_z >= other.min_z)
    }

    fn extend(&mut self, other: Self) {
        self.min_x = self.min_x.min(other.min_x);
        self.min_y = self.min_y.min(other.min_y);
        self.min_z = self.min_z.min(other.min_z);
        self.max_x = self.max_x.max(other.max_x);
        self.max_y = self.max_y.max(other.max_y);
        self.max_z = self.max_z.max(other.max_z);
    }

    fn distance_squared_to(self, point: Point3D) -> f64 {
        let dx = axis_distance(point.x, self.min_x, self.max_x);
        let dy = axis_distance(point.y, self.min_y, self.max_y);
        let dz = axis_distance(point.z, self.min_z, self.max_z);
        dx * dx + dy * dy + dz * dz
    }
}

#[derive(Clone, Copy, Debug)]
struct Point3D {
    x: f64,
    y: f64,
    z: f64,
}

impl Point3D {
    const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

#[derive(Clone, Copy, Debug)]
enum DatasetKind {
    PlanarXY,
    Uniform,
    Clustered,
    FlatZ,
    Degenerate,
}

#[derive(Clone, Copy, Debug)]
enum SortKey3D {
    Morton,
    Hilbert,
}

#[derive(Clone, Copy, Debug, Default)]
struct BuildMetrics {
    sort_time: Duration,
    pack_time: Duration,
}

#[derive(Clone, Copy, Debug, Default)]
struct SearchMetrics {
    elapsed: Duration,
    visited_bounds: usize,
    results: usize,
}

#[derive(Debug)]
struct PrototypeIndex3D {
    node_size: usize,
    num_items: usize,
    level_bounds: Vec<usize>,
    boxes: Vec<Bounds3D>,
    indices: Vec<usize>,
}

impl PrototypeIndex3D {
    fn build(items: &[Bounds3D], node_size: usize, sort_key: SortKey3D) -> (Self, BuildMetrics) {
        let node_size = node_size.clamp(2, 65_535);
        let num_items = items.len();
        let mut level_bounds = Vec::new();
        let mut num_nodes = num_items;
        let mut n = num_items;
        level_bounds.push(n);
        if num_items > 0 {
            loop {
                n = n.div_ceil(node_size);
                num_nodes += n;
                level_bounds.push(num_nodes);
                if n == 1 {
                    break;
                }
            }
        }

        if num_items == 0 {
            return (
                Self {
                    node_size,
                    num_items,
                    level_bounds,
                    boxes: Vec::new(),
                    indices: Vec::new(),
                },
                BuildMetrics::default(),
            );
        }

        let tree_extent = extent(items);
        let sort_start = Instant::now();
        let mut order: Vec<(u64, usize)> = (0..num_items)
            .map(|i| (encode_sort_key(tree_extent, items[i], sort_key), i))
            .collect();
        order.sort_unstable_by_key(|&(key, _)| key);
        let sort_time = sort_start.elapsed();

        let pack_start = Instant::now();
        let mut boxes = vec![Bounds3D::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0); num_nodes];
        let mut indices = vec![0usize; num_nodes];

        for (slot, &(_, original)) in order.iter().enumerate() {
            boxes[slot] = items[original];
            indices[slot] = original;
        }

        let mut read_pos = 0usize;
        let mut write_pos = num_items;
        for &level_end in &level_bounds[..level_bounds.len() - 1] {
            while read_pos < level_end {
                let node_index = read_pos;
                let mut node_bounds = empty_bounds();
                let mut children = 0usize;
                while children < node_size && read_pos < level_end {
                    node_bounds.extend(boxes[read_pos]);
                    read_pos += 1;
                    children += 1;
                }
                boxes[write_pos] = node_bounds;
                indices[write_pos] = node_index;
                write_pos += 1;
            }
        }
        let pack_time = pack_start.elapsed();

        (
            Self {
                node_size,
                num_items,
                level_bounds,
                boxes,
                indices,
            },
            BuildMetrics {
                sort_time,
                pack_time,
            },
        )
    }

    fn search(&self, query: Bounds3D) -> Vec<usize> {
        let mut results = Vec::new();
        let mut stack = Vec::new();
        self.search_into(query, &mut results, &mut stack);
        results
    }

    fn search_into(
        &self,
        query: Bounds3D,
        results: &mut Vec<usize>,
        stack: &mut Vec<(usize, usize)>,
    ) -> SearchMetrics {
        let start = Instant::now();
        let mut visited_bounds = 0usize;
        results.clear();
        stack.clear();
        if self.num_items == 0 {
            return SearchMetrics {
                elapsed: start.elapsed(),
                visited_bounds,
                results: 0,
            };
        }

        let mut node_index = self.boxes.len() - 1;
        let mut level = self.level_bounds.len() - 1;

        loop {
            let end = (node_index + self.node_size).min(self.level_bounds[level]);
            let is_leaf = node_index < self.num_items;
            for pos in node_index..end {
                visited_bounds += 1;
                if !self.boxes[pos].overlaps(query) {
                    continue;
                }
                let index = self.indices[pos];
                if is_leaf {
                    results.push(index);
                } else {
                    stack.push((index, level - 1));
                }
            }

            if let Some((next_node, next_level)) = stack.pop() {
                node_index = next_node;
                level = next_level;
            } else {
                return SearchMetrics {
                    elapsed: start.elapsed(),
                    visited_bounds,
                    results: results.len(),
                };
            }
        }
    }

    fn neighbors(&self, point: Point3D, max_results: usize, max_distance: f64) -> Vec<usize> {
        if max_results == 0 || max_distance.is_nan() || max_distance.is_sign_negative() {
            return Vec::new();
        }
        if self.num_items == 0 {
            return Vec::new();
        }

        let max_distance_squared = max_distance * max_distance;
        let mut results = Vec::with_capacity(max_results);
        let mut queue = BinaryHeap::new();
        let root = self.boxes.len() - 1;
        queue.push(QueueEntry::node(
            root,
            self.level_bounds.len() - 1,
            self.boxes[root].distance_squared_to(point),
        ));

        while let Some(entry) = queue.pop() {
            if entry.distance_squared > max_distance_squared {
                break;
            }

            match entry.kind {
                QueueEntryKind::Item { original_index } => {
                    results.push(original_index);
                    if results.len() == max_results {
                        break;
                    }
                }
                QueueEntryKind::Node { node_index, level } => {
                    let end = (node_index + self.node_size).min(self.level_bounds[level]);
                    let child_level = level.saturating_sub(1);
                    let entries_are_items = level == 0;
                    for pos in node_index..end {
                        let distance_squared = self.boxes[pos].distance_squared_to(point);
                        if distance_squared > max_distance_squared {
                            continue;
                        }
                        let index = self.indices[pos];
                        if entries_are_items {
                            queue.push(QueueEntry::item(index, distance_squared));
                        } else {
                            queue.push(QueueEntry::node(index, child_level, distance_squared));
                        }
                    }
                }
            }
        }

        results
    }
}

#[derive(Clone, Copy, Debug)]
struct QueueEntry {
    distance_squared: f64,
    tie_breaker: usize,
    kind: QueueEntryKind,
}

impl QueueEntry {
    const fn item(original_index: usize, distance_squared: f64) -> Self {
        Self {
            distance_squared,
            tie_breaker: original_index,
            kind: QueueEntryKind::Item { original_index },
        }
    }

    const fn node(node_index: usize, level: usize, distance_squared: f64) -> Self {
        Self {
            distance_squared,
            tie_breaker: node_index,
            kind: QueueEntryKind::Node { node_index, level },
        }
    }
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance_squared.to_bits() == other.distance_squared.to_bits()
            && self.tie_breaker == other.tie_breaker
            && self.kind == other.kind
    }
}

impl Eq for QueueEntry {}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance_squared
            .total_cmp(&self.distance_squared)
            .then_with(|| other.tie_breaker.cmp(&self.tie_breaker))
            .then_with(|| other.kind.cmp(&self.kind))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum QueueEntryKind {
    Node { node_index: usize, level: usize },
    Item { original_index: usize },
}

#[test]
#[ignore = "research-only: sanity-checks temporary 3D sort-key encoders"]
fn bounds3d_sort_keys_are_injective_on_small_grid() {
    for sort_key in [SortKey3D::Morton, SortKey3D::Hilbert] {
        let mut seen = std::collections::BTreeSet::new();
        for x in 0..8 {
            for y in 0..8 {
                for z in 0..8 {
                    let key = match sort_key {
                        SortKey3D::Morton => morton3(x, y, z),
                        SortKey3D::Hilbert => hilbert3(x, y, z),
                    };
                    assert!(
                        seen.insert(key),
                        "{sort_key:?} duplicate key {key} for ({x}, {y}, {z})"
                    );
                }
            }
        }
        assert_eq!(seen.len(), 512, "{sort_key:?}");
    }
}

#[test]
#[ignore = "research-only: validates the temporary 3D prototype against brute force"]
fn bounds3d_sort_keys_correctness_against_bruteforce() {
    let datasets = [
        DatasetKind::PlanarXY,
        DatasetKind::Uniform,
        DatasetKind::Clustered,
        DatasetKind::FlatZ,
        DatasetKind::Degenerate,
    ];

    for dataset in datasets {
        let items = make_dataset(dataset, 512, 0x3D00);
        let queries = make_queries(dataset, 48, 0x3D01);
        let points = make_points(dataset, 48, 0x3D02);

        for node_size in [8, 16, 32] {
            for sort_key in [SortKey3D::Morton, SortKey3D::Hilbert] {
                let (index, _) = PrototypeIndex3D::build(&items, node_size, sort_key);

                for &query in &queries {
                    let mut actual = index.search(query);
                    let mut expected = brute_force_search(&items, query);
                    actual.sort_unstable();
                    expected.sort_unstable();
                    assert_eq!(
                        actual, expected,
                        "{dataset:?}, {sort_key:?}, node_size={node_size}"
                    );
                }

                for &point in &points {
                    assert_eq!(
                        index.neighbors(point, 1, f64::INFINITY),
                        brute_force_neighbors(&items, point, 1, f64::INFINITY),
                        "{dataset:?}, {sort_key:?}, node_size={node_size}, top-1"
                    );
                    assert_eq!(
                        index.neighbors(point, 10, f64::INFINITY),
                        brute_force_neighbors(&items, point, 10, f64::INFINITY),
                        "{dataset:?}, {sort_key:?}, node_size={node_size}, top-10"
                    );
                    assert_eq!(
                        index.neighbors(point, 10, 80.0),
                        brute_force_neighbors(&items, point, 10, 80.0),
                        "{dataset:?}, {sort_key:?}, node_size={node_size}, finite radius"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "research-only: prints 3D sort-key and node-size layout metrics"]
fn bounds3d_sort_key_node_size_survey() {
    println!(
        "dataset,sort_key,node_size,items,queries,sort_ms,pack_ms,search_ms,avg_visited,avg_hits,knn1_ms,knn10_ms,knn10_r80_ms"
    );

    for dataset in [
        DatasetKind::PlanarXY,
        DatasetKind::Uniform,
        DatasetKind::Clustered,
        DatasetKind::FlatZ,
        DatasetKind::Degenerate,
    ] {
        let items = make_dataset(dataset, RESEARCH_ITEMS, 0xA11CE);
        let queries = make_queries(dataset, RESEARCH_QUERIES, 0xB0B);
        let points = make_points(dataset, RESEARCH_QUERIES, 0xCAFE);

        for sort_key in [SortKey3D::Morton, SortKey3D::Hilbert] {
            for node_size in [8, 16, 32] {
                let (index, build) = PrototypeIndex3D::build(&items, node_size, sort_key);
                let mut results = Vec::new();
                let mut stack = Vec::new();
                let mut search = SearchMetrics::default();

                for &query in &queries {
                    let metrics = index.search_into(query, &mut results, &mut stack);
                    search.elapsed += metrics.elapsed;
                    search.visited_bounds += metrics.visited_bounds;
                    search.results += metrics.results;
                }

                let knn1 = time_neighbors(&index, &points, 1, f64::INFINITY);
                let knn10 = time_neighbors(&index, &points, 10, f64::INFINITY);
                let knn10_r80 = time_neighbors(&index, &points, 10, 80.0);

                println!(
                    "{dataset:?},{sort_key:?},{node_size},{},{},{:.3},{:.3},{:.3},{:.2},{:.2},{:.3},{:.3},{:.3}",
                    items.len(),
                    queries.len(),
                    millis(build.sort_time),
                    millis(build.pack_time),
                    millis(search.elapsed),
                    search.visited_bounds as f64 / queries.len() as f64,
                    search.results as f64 / queries.len() as f64,
                    millis(knn1),
                    millis(knn10),
                    millis(knn10_r80),
                );
            }
        }
    }
}

#[test]
#[ignore = "research-only: compares current 2D Hilbert encoding with temporary 3D Hilbert encoding"]
fn hilbert2d_vs_hilbert3d_encode_survey() {
    let mut rng = StdRng::seed_from_u64(0x2D3D);
    let coords2d: Vec<(u16, u16)> = (0..ENCODE_ITEMS)
        .map(|_| {
            (
                rng.random_range(0..=u16::MAX),
                rng.random_range(0..=u16::MAX),
            )
        })
        .collect();
    let coords3d: Vec<(u32, u32, u32)> = (0..ENCODE_ITEMS)
        .map(|_| {
            (
                rng.random_range(0..=HILBERT_AXIS_MAX),
                rng.random_range(0..=HILBERT_AXIS_MAX),
                rng.random_range(0..=HILBERT_AXIS_MAX),
            )
        })
        .collect();

    println!("encoder,items,total_ms,ns_per_key,checksum");
    let (elapsed2d, checksum2d) = time_hilbert2d_encode(&coords2d);
    let (elapsed3d, checksum3d) = time_hilbert3d_encode(&coords3d);
    println!(
        "Hilbert2D,{},{:.3},{:.2},{}",
        coords2d.len(),
        millis(elapsed2d),
        elapsed2d.as_nanos() as f64 / coords2d.len() as f64,
        checksum2d
    );
    println!(
        "Hilbert3D,{},{:.3},{:.2},{}",
        coords3d.len(),
        millis(elapsed3d),
        elapsed3d.as_nanos() as f64 / coords3d.len() as f64,
        checksum3d
    );
}

#[test]
#[ignore = "research-only: compares production 2D Hilbert index with temporary 3D Hilbert prototype"]
fn hilbert2d_vs_hilbert3d_index_survey() {
    println!(
        "dataset,dimension,node_size,items,queries,build_ms,search_ms,avg_visited,avg_hits,knn1_ms,knn10_ms,knn10_r80_ms"
    );

    for dataset in [
        DatasetKind::PlanarXY,
        DatasetKind::Uniform,
        DatasetKind::Clustered,
        DatasetKind::FlatZ,
        DatasetKind::Degenerate,
    ] {
        let items3d = make_dataset(dataset, RESEARCH_ITEMS, 0xA11CE);
        let queries3d = make_queries(dataset, RESEARCH_QUERIES, 0xB0B);
        let points3d = make_points(dataset, RESEARCH_QUERIES, 0xCAFE);
        let items2d = project_items_2d(&items3d);
        let queries2d = project_queries_2d(&queries3d);
        let points2d = project_points_2d(&points3d);

        for node_size in [8, 16] {
            let (index2d, build2d) = build_2d_hilbert(&items2d, node_size);
            let search2d = time_search_2d(&index2d, &queries2d);
            let knn1_2d = time_neighbors_2d(&index2d, &points2d, 1, f64::INFINITY);
            let knn10_2d = time_neighbors_2d(&index2d, &points2d, 10, f64::INFINITY);
            let knn10_r80_2d = time_neighbors_2d(&index2d, &points2d, 10, 80.0);

            println!(
                "{dataset:?},2D,{node_size},{},{},{:.3},{:.3},{:.2},{:.2},{:.3},{:.3},{:.3}",
                items2d.len(),
                queries2d.len(),
                millis(build2d),
                millis(search2d.elapsed),
                search2d.visited_bounds as f64 / queries2d.len() as f64,
                search2d.results as f64 / queries2d.len() as f64,
                millis(knn1_2d),
                millis(knn10_2d),
                millis(knn10_r80_2d),
            );

            let build3d_start = Instant::now();
            let (index3d, _) = PrototypeIndex3D::build(&items3d, node_size, SortKey3D::Hilbert);
            let build3d = build3d_start.elapsed();
            let mut results = Vec::new();
            let mut stack = Vec::new();
            let mut search3d = SearchMetrics::default();
            for &query in &queries3d {
                let metrics = index3d.search_into(query, &mut results, &mut stack);
                search3d.elapsed += metrics.elapsed;
                search3d.visited_bounds += metrics.visited_bounds;
                search3d.results += metrics.results;
            }
            let knn1_3d = time_neighbors(&index3d, &points3d, 1, f64::INFINITY);
            let knn10_3d = time_neighbors(&index3d, &points3d, 10, f64::INFINITY);
            let knn10_r80_3d = time_neighbors(&index3d, &points3d, 10, 80.0);

            println!(
                "{dataset:?},3D,{node_size},{},{},{:.3},{:.3},{:.2},{:.2},{:.3},{:.3},{:.3}",
                items3d.len(),
                queries3d.len(),
                millis(build3d),
                millis(search3d.elapsed),
                search3d.visited_bounds as f64 / queries3d.len() as f64,
                search3d.results as f64 / queries3d.len() as f64,
                millis(knn1_3d),
                millis(knn10_3d),
                millis(knn10_r80_3d),
            );
        }
    }
}

fn time_hilbert2d_encode(coords: &[(u16, u16)]) -> (Duration, u64) {
    let start = Instant::now();
    let mut checksum = 0u64;
    for &(x, y) in coords {
        checksum ^= u64::from(hilbert2d(black_box(x), black_box(y)));
    }
    (start.elapsed(), black_box(checksum))
}

fn time_hilbert3d_encode(coords: &[(u32, u32, u32)]) -> (Duration, u64) {
    let start = Instant::now();
    let mut checksum = 0u64;
    for &(x, y, z) in coords {
        checksum ^= hilbert3(black_box(x), black_box(y), black_box(z));
    }
    (start.elapsed(), black_box(checksum))
}

fn build_2d_hilbert(items: &[Bounds2D], node_size: usize) -> (Index2D, Duration) {
    let start = Instant::now();
    let mut builder = Index2DBuilder::new(items.len()).node_size(node_size);
    for &bounds in items {
        builder.add(bounds);
    }
    let index = builder.finish().unwrap();
    (index, start.elapsed())
}

fn time_search_2d(index: &Index2D, queries: &[Bounds2D]) -> SearchMetrics {
    let start = Instant::now();
    let mut visited_bounds = 0usize;
    let mut results = 0usize;
    for &query in queries {
        let (result_count, visited_count) = index.search_visited(query);
        results += result_count;
        visited_bounds += visited_count;
    }
    SearchMetrics {
        elapsed: start.elapsed(),
        visited_bounds,
        results,
    }
}

fn time_neighbors_2d(
    index: &Index2D,
    points: &[Point2D],
    max_results: usize,
    max_distance: f64,
) -> Duration {
    let start = Instant::now();
    let mut total = 0usize;
    for &point in points {
        total += index
            .neighbors_within(point, max_results, max_distance)
            .len();
    }
    assert_ne!(total, usize::MAX);
    start.elapsed()
}

fn time_neighbors(
    index: &PrototypeIndex3D,
    points: &[Point3D],
    max_results: usize,
    max_distance: f64,
) -> Duration {
    let start = Instant::now();
    let mut total = 0usize;
    for &point in points {
        total += index.neighbors(point, max_results, max_distance).len();
    }
    assert_ne!(total, usize::MAX);
    start.elapsed()
}

fn project_items_2d(items: &[Bounds3D]) -> Vec<Bounds2D> {
    items
        .iter()
        .map(|bounds| Bounds2D::new(bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y))
        .collect()
}

fn project_queries_2d(queries: &[Bounds3D]) -> Vec<Bounds2D> {
    queries
        .iter()
        .map(|bounds| Bounds2D::new(bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y))
        .collect()
}

fn project_points_2d(points: &[Point3D]) -> Vec<Point2D> {
    points
        .iter()
        .map(|point| Point2D::new(point.x, point.y))
        .collect()
}

fn make_dataset(kind: DatasetKind, n: usize, seed: u64) -> Vec<Bounds3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    match kind {
        DatasetKind::PlanarXY => (0..n)
            .map(|_| {
                let xy = random_box(&mut rng, 0.0..10_000.0, 0.0..10_000.0, 0.0..1.0, 1.0..40.0);
                Bounds3D::new(xy.min_x, xy.min_y, 0.0, xy.max_x, xy.max_y, 0.0)
            })
            .collect(),
        DatasetKind::Uniform => (0..n)
            .map(|_| {
                random_box(
                    &mut rng,
                    0.0..10_000.0,
                    0.0..10_000.0,
                    0.0..10_000.0,
                    1.0..40.0,
                )
            })
            .collect(),
        DatasetKind::Clustered => (0..n)
            .map(|i| {
                let cluster = (i % 8) as f64;
                let base_x = 1_000.0 + cluster * 900.0;
                let base_y = 1_000.0 + (cluster % 4.0) * 1_200.0;
                let base_z = 1_000.0 + (cluster / 2.0).floor() * 700.0;
                random_box(
                    &mut rng,
                    base_x..base_x + 250.0,
                    base_y..base_y + 250.0,
                    base_z..base_z + 250.0,
                    1.0..30.0,
                )
            })
            .collect(),
        DatasetKind::FlatZ => (0..n)
            .map(|_| random_box(&mut rng, 0.0..10_000.0, 0.0..10_000.0, 0.0..20.0, 1.0..35.0))
            .collect(),
        DatasetKind::Degenerate => (0..n)
            .map(|_| {
                let x: f64 = rng.random_range(0.0..10_000.0);
                let y: f64 = rng.random_range(0.0..10_000.0);
                let z: f64 = rng.random_range(0.0..10_000.0);
                Bounds3D::new(x, y, z, x, y, z)
            })
            .collect(),
    }
}

fn make_queries(kind: DatasetKind, n: usize, seed: u64) -> Vec<Bounds3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| match kind {
            DatasetKind::PlanarXY => {
                let xy = random_box(
                    &mut rng,
                    0.0..10_000.0,
                    0.0..10_000.0,
                    0.0..1.0,
                    50.0..300.0,
                );
                Bounds3D::new(xy.min_x, xy.min_y, 0.0, xy.max_x, xy.max_y, 0.0)
            }
            DatasetKind::Clustered => random_box(
                &mut rng,
                800.0..8_500.0,
                800.0..6_000.0,
                800.0..4_000.0,
                50.0..300.0,
            ),
            DatasetKind::FlatZ => random_box(
                &mut rng,
                0.0..10_000.0,
                0.0..10_000.0,
                0.0..20.0,
                30.0..250.0,
            ),
            DatasetKind::Degenerate | DatasetKind::Uniform => random_box(
                &mut rng,
                0.0..10_000.0,
                0.0..10_000.0,
                0.0..10_000.0,
                50.0..300.0,
            ),
        })
        .collect()
}

fn make_points(kind: DatasetKind, n: usize, seed: u64) -> Vec<Point3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| match kind {
            DatasetKind::PlanarXY => Point3D::new(
                rng.random_range(0.0..10_000.0),
                rng.random_range(0.0..10_000.0),
                0.0,
            ),
            DatasetKind::Clustered => Point3D::new(
                rng.random_range(800.0..8_500.0),
                rng.random_range(800.0..6_000.0),
                rng.random_range(800.0..4_000.0),
            ),
            DatasetKind::FlatZ => Point3D::new(
                rng.random_range(0.0..10_000.0),
                rng.random_range(0.0..10_000.0),
                rng.random_range(0.0..20.0),
            ),
            DatasetKind::Degenerate | DatasetKind::Uniform => Point3D::new(
                rng.random_range(0.0..10_000.0),
                rng.random_range(0.0..10_000.0),
                rng.random_range(0.0..10_000.0),
            ),
        })
        .collect()
}

fn random_box(
    rng: &mut StdRng,
    x_range: std::ops::Range<f64>,
    y_range: std::ops::Range<f64>,
    z_range: std::ops::Range<f64>,
    size_range: std::ops::Range<f64>,
) -> Bounds3D {
    let x = rng.random_range(x_range);
    let y = rng.random_range(y_range);
    let z = rng.random_range(z_range);
    let dx = rng.random_range(size_range.clone());
    let dy = rng.random_range(size_range.clone());
    let dz = rng.random_range(size_range);
    Bounds3D::new(x, y, z, x + dx, y + dy, z + dz)
}

fn brute_force_search(items: &[Bounds3D], query: Bounds3D) -> Vec<usize> {
    items
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, bounds)| bounds.overlaps(query).then_some(index))
        .collect()
}

fn brute_force_neighbors(
    items: &[Bounds3D],
    point: Point3D,
    max_results: usize,
    max_distance: f64,
) -> Vec<usize> {
    if max_results == 0 || max_distance.is_nan() || max_distance.is_sign_negative() {
        return Vec::new();
    }
    let max_distance_squared = max_distance * max_distance;
    let mut pairs: Vec<(usize, f64)> = items
        .iter()
        .copied()
        .enumerate()
        .map(|(index, bounds)| (index, bounds.distance_squared_to(point)))
        .filter(|&(_, distance_squared)| distance_squared <= max_distance_squared)
        .collect();
    pairs.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    pairs
        .into_iter()
        .take(max_results)
        .map(|(index, _)| index)
        .collect()
}

fn encode_sort_key(extent: Bounds3D, bounds: Bounds3D, sort_key: SortKey3D) -> u64 {
    match sort_key {
        SortKey3D::Morton => {
            let x = normalize_center(
                bounds.min_x,
                bounds.max_x,
                extent.min_x,
                extent.max_x,
                MORTON_AXIS_MAX,
            );
            let y = normalize_center(
                bounds.min_y,
                bounds.max_y,
                extent.min_y,
                extent.max_y,
                MORTON_AXIS_MAX,
            );
            let z = normalize_center(
                bounds.min_z,
                bounds.max_z,
                extent.min_z,
                extent.max_z,
                MORTON_AXIS_MAX,
            );
            morton3(x, y, z)
        }
        SortKey3D::Hilbert => {
            let x = normalize_center(
                bounds.min_x,
                bounds.max_x,
                extent.min_x,
                extent.max_x,
                HILBERT_AXIS_MAX,
            );
            let y = normalize_center(
                bounds.min_y,
                bounds.max_y,
                extent.min_y,
                extent.max_y,
                HILBERT_AXIS_MAX,
            );
            let z = normalize_center(
                bounds.min_z,
                bounds.max_z,
                extent.min_z,
                extent.max_z,
                HILBERT_AXIS_MAX,
            );
            hilbert3(x, y, z)
        }
    }
}

fn extent(items: &[Bounds3D]) -> Bounds3D {
    let mut extent = empty_bounds();
    for &bounds in items {
        extent.extend(bounds);
    }
    extent
}

fn empty_bounds() -> Bounds3D {
    Bounds3D::new(
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    )
}

fn normalize_center(min: f64, max: f64, extent_min: f64, extent_max: f64, axis_max: u32) -> u32 {
    let width = extent_max - extent_min;
    if width <= 0.0 || !width.is_finite() {
        return 0;
    }

    let normalized = ((0.5 * (min + max) - extent_min) / width) * f64::from(axis_max);
    if normalized.is_nan() || normalized <= 0.0 {
        0
    } else if normalized >= f64::from(axis_max) {
        axis_max
    } else {
        normalized as u32
    }
}

#[inline(always)]
fn morton3(x: u32, y: u32, z: u32) -> u64 {
    split_by_3(u64::from(x)) | (split_by_3(u64::from(y)) << 1) | (split_by_3(u64::from(z)) << 2)
}

#[inline(always)]
fn hilbert3(x: u32, y: u32, z: u32) -> u64 {
    hilbert3_state_machine(x, y, z)
}

#[inline(always)]
fn interleave3_msb_order(x: u32, y: u32, z: u32) -> u64 {
    split_by_3(u64::from(z)) | (split_by_3(u64::from(y)) << 1) | (split_by_3(u64::from(x)) << 2)
}

#[inline(always)]
fn hilbert3_state_machine(x: u32, y: u32, z: u32) -> u64 {
    let mut index = 0u64;
    let mut state = 0usize;

    for shift in (0..HILBERT_BITS_PER_AXIS).rev() {
        let m = (((x >> shift) & 1) << 2) | (((y >> shift) & 1) << 1) | ((z >> shift) & 1);
        let entry = HILBERT3_LUT[state * 8 + m as usize];
        index = (index << 3) | u64::from(entry & 7);
        state = (entry >> 3) as usize;
    }

    index
}

const fn build_hilbert3_lut() -> [u8; 192] {
    let mut table = [0u8; 192];
    let mut state = 0usize;
    while state < 24 {
        let c = (state & 7) as u32;
        let n = (state / 8) as u32;
        let mut m = 0u32;
        while m < 8 {
            let gray = rotate_right_3(c ^ m, n);
            let i = gray_to_integer_3(gray);
            let without_high_bit = gray & 0b011;
            let next_rotation = if without_high_bit == 0 {
                1
            } else if (without_high_bit & 1) != 0 {
                2
            } else {
                3
            };
            let transform = if i == 0 {
                0
            } else {
                let low_bit = i & 0u32.wrapping_sub(i);
                gray ^ (low_bit | 1)
            };
            let next_c = c ^ rotate_left_3(transform, n);
            let next_n = (n + next_rotation) % 3;
            let next_state = next_n * 8 + next_c;
            table[state * 8 + m as usize] = ((next_state as u8) << 3) | (i as u8);
            m += 1;
        }
        state += 1;
    }
    table
}

const fn rotate_left_3(value: u32, shift: u32) -> u32 {
    match shift {
        0 => value & 7,
        1 => ((value << 1) | (value >> 2)) & 7,
        _ => ((value << 2) | (value >> 1)) & 7,
    }
}

const fn rotate_right_3(value: u32, shift: u32) -> u32 {
    match shift {
        0 => value & 7,
        1 => ((value >> 1) | (value << 2)) & 7,
        _ => ((value >> 2) | (value << 1)) & 7,
    }
}

const fn gray_to_integer_3(mut gray: u32) -> u32 {
    gray ^= gray >> 1;
    gray ^= gray >> 2;
    gray & 7
}

#[inline(always)]
fn hilbert3_axes_to_transpose(mut x: u32, mut y: u32, mut z: u32) -> (u32, u32, u32) {
    let highest_bit = 1 << (HILBERT_BITS_PER_AXIS - 1);

    let mut q = highest_bit;
    while q > 1 {
        let p = q - 1;
        let x_bit = mask_from_nonzero(x & q);
        x ^= p & x_bit;
        conditional_hilbert_exchange(&mut x, &mut y, p, q);
        conditional_hilbert_exchange(&mut x, &mut z, p, q);
        q >>= 1;
    }

    y ^= x;
    z ^= y;

    let mut t = 0;
    q = highest_bit;
    while q > 1 {
        t ^= (q - 1) & mask_from_nonzero(z & q);
        q >>= 1;
    }

    x ^= t;
    y ^= t;
    z ^= t;
    (x, y, z)
}

#[inline(always)]
fn conditional_hilbert_exchange(x: &mut u32, axis: &mut u32, p: u32, q: u32) {
    let axis_bit = mask_from_nonzero(*axis & q);
    let swap = (*x ^ *axis) & p;
    let x_delta = (p & axis_bit) | (swap & !axis_bit);
    let axis_delta = swap & !axis_bit;
    *x ^= x_delta;
    *axis ^= axis_delta;
}

#[inline(always)]
fn mask_from_nonzero(value: u32) -> u32 {
    0u32.wrapping_sub(u32::from(value != 0))
}

#[inline(always)]
fn split_by_3(mut value: u64) -> u64 {
    value &= 0x1f_ffff;
    value = (value | (value << 32)) & 0x001f_0000_0000_ffff;
    value = (value | (value << 16)) & 0x001f_0000_ff00_00ff;
    value = (value | (value << 8)) & 0x100f_00f0_0f00_f00f;
    value = (value | (value << 4)) & 0x10c3_0c30_c30c_30c3;
    value = (value | (value << 2)) & 0x1249_2492_4924_9249;
    value
}

fn axis_distance(point: f64, min: f64, max: f64) -> f64 {
    if point < min {
        min - point
    } else if point > max {
        point - max
    } else {
        0.0
    }
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
