#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const MORTON_BITS_PER_AXIS: u32 = 21;
const MORTON_AXIS_MAX: u32 = (1 << MORTON_BITS_PER_AXIS) - 1;
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

fn make_dataset(kind: DatasetKind, n: usize, seed: u64) -> Vec<Bounds3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    match kind {
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
    let x = normalize_center(bounds.min_x, bounds.max_x, extent.min_x, extent.max_x);
    let y = normalize_center(bounds.min_y, bounds.max_y, extent.min_y, extent.max_y);
    let z = normalize_center(bounds.min_z, bounds.max_z, extent.min_z, extent.max_z);
    match sort_key {
        SortKey3D::Morton => morton3(x, y, z),
        SortKey3D::Hilbert => hilbert3(x, y, z),
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

fn normalize_center(min: f64, max: f64, extent_min: f64, extent_max: f64) -> u32 {
    let width = extent_max - extent_min;
    if width <= 0.0 || !width.is_finite() {
        return 0;
    }

    let normalized = ((0.5 * (min + max) - extent_min) / width) * f64::from(MORTON_AXIS_MAX);
    if normalized.is_nan() || normalized <= 0.0 {
        0
    } else if normalized >= f64::from(MORTON_AXIS_MAX) {
        MORTON_AXIS_MAX
    } else {
        normalized as u32
    }
}

fn morton3(x: u32, y: u32, z: u32) -> u64 {
    split_by_3(u64::from(x)) | (split_by_3(u64::from(y)) << 1) | (split_by_3(u64::from(z)) << 2)
}

fn hilbert3(x: u32, y: u32, z: u32) -> u64 {
    let mut axes = [x, y, z];
    hilbert_axes_to_transpose(&mut axes);

    let mut key = 0u64;
    for bit in (0..MORTON_BITS_PER_AXIS).rev() {
        key = (key << 1) | u64::from((axes[0] >> bit) & 1);
        key = (key << 1) | u64::from((axes[1] >> bit) & 1);
        key = (key << 1) | u64::from((axes[2] >> bit) & 1);
    }
    key
}

fn hilbert_axes_to_transpose(axes: &mut [u32; 3]) {
    let highest_bit = 1 << (MORTON_BITS_PER_AXIS - 1);

    let mut q = highest_bit;
    while q > 1 {
        let p = q - 1;
        for i in 0..axes.len() {
            if (axes[i] & q) != 0 {
                axes[0] ^= p;
            } else {
                let t = (axes[0] ^ axes[i]) & p;
                axes[0] ^= t;
                axes[i] ^= t;
            }
        }
        q >>= 1;
    }

    for i in 1..axes.len() {
        axes[i] ^= axes[i - 1];
    }

    let mut t = 0;
    q = highest_bit;
    while q > 1 {
        if (axes[axes.len() - 1] & q) != 0 {
            t ^= q - 1;
        }
        q >>= 1;
    }

    for axis in axes {
        *axis ^= t;
    }
}

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
