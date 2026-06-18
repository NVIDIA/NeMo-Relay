// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Geometric summaries of data blocks and hierarchical aggregation trees.

use libm::sqrt;

/// Maximum number of blocks retained at the finest level.
const MAX_BLOCKS: usize = 128;

/// Threshold below which a vector norm is treated as zero to avoid division
/// by very small numbers.
const NORM_EPSILON: f64 = 1e-10;

/// Variance threshold below which a block is considered tightly clustered.
const LOW_VARIANCE_THRESHOLD: f64 = 0.1;

/// Concentration threshold above which a block is considered highly aligned.
const HIGH_CONCENTRATION_THRESHOLD: f64 = 0.9;

/// Compression ratio assigned to strategies that encode blocks compactly.
const COMPACT_COMPRESSION_RATIO: f64 = 4.0;

/// Compression ratio assigned when no compression is applied.
const NO_COMPRESSION_RATIO: f64 = 1.0;

/// Geometric summary of a block of `D`-dimensional points.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockMetadata<const D: usize> {
    /// Centroid (arithmetic mean) of the points in the block.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_arrays"))]
    pub centroid: [f64; D],
    /// Maximum distance from the centroid to any point in the block.
    pub radius: f64,
    /// Variance of distances from the centroid.
    pub variance: f64,
    /// Average cosine alignment of points with the centroid.
    pub concentration: f64,
    /// Number of points summarized by this block.
    pub count: usize,
}

impl<const D: usize> Default for BlockMetadata<D> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<const D: usize> BlockMetadata<D> {
    /// Return an empty block with all fields set to zero.
    pub const fn empty() -> Self {
        Self {
            centroid: [0.0; D],
            radius: 0.0,
            variance: 0.0,
            concentration: 0.0,
            count: 0,
        }
    }

    /// Compute metadata from a slice of `D`-dimensional points.
    pub fn from_points(points: &[[f64; D]]) -> Self {
        if points.is_empty() {
            return Self::empty();
        }

        let n = points.len();
        let mut centroid = [0.0f64; D];
        for point in points {
            for d in 0..D {
                centroid[d] += point[d];
            }
        }
        for val in centroid.iter_mut() {
            *val /= n as f64;
        }

        let centroid_norm = vector_norm(&centroid);
        let mut max_dist = 0.0f64;
        let mut sum_dist = 0.0f64;
        let mut sum_dist_sq = 0.0f64;
        let mut sum_cosine = 0.0f64;

        for point in points {
            let dist = l2_distance(&centroid, point);
            max_dist = max_dist.max(dist);
            sum_dist += dist;
            sum_dist_sq += dist * dist;

            if centroid_norm > NORM_EPSILON {
                let point_norm = vector_norm(point);
                if point_norm > NORM_EPSILON {
                    let dot = dot_product(&centroid, point);
                    sum_cosine += dot / (centroid_norm * point_norm);
                }
            }
        }

        let mean_dist = sum_dist / n as f64;
        let variance = (sum_dist_sq / n as f64) - (mean_dist * mean_dist);

        Self {
            centroid,
            radius: max_dist,
            variance: variance.max(0.0),
            concentration: sum_cosine / n as f64,
            count: n,
        }
    }

    /// Return an upper bound on the dot-product score with `query`.
    ///
    /// The bound follows from the Cauchy-Schwarz inequality:
    /// `q · p ≤ ||q|| * (||centroid|| + radius)` for any point `p` in the
    /// block.
    pub fn upper_bound_score(&self, query: &[f64; D]) -> f64 {
        let q_norm = vector_norm(query);
        let c_norm = vector_norm(&self.centroid);
        q_norm * (c_norm + self.radius)
    }

    /// Return true if the block can be pruned because its upper-bound score
    /// is below `threshold`.
    pub fn can_prune(&self, query: &[f64; D], threshold: f64) -> bool {
        self.upper_bound_score(query) < threshold
    }
}

/// Strategy for compressing a block based on its geometric properties.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressionStrategy {
    /// Store the centroid and delta-encode residuals.
    CentroidDelta,
    /// Aggressive 4-bit quantization for highly concentrated blocks.
    Int4Quantize,
    /// Keep full precision for dispersed blocks.
    FullPrecision,
}

/// Select a compression strategy for a block from its metadata.
pub fn select_compression<const D: usize>(meta: &BlockMetadata<D>) -> CompressionStrategy {
    if meta.variance < LOW_VARIANCE_THRESHOLD {
        CompressionStrategy::CentroidDelta
    } else if meta.concentration > HIGH_CONCENTRATION_THRESHOLD {
        CompressionStrategy::Int4Quantize
    } else {
        CompressionStrategy::FullPrecision
    }
}

/// Estimate the compression ratio achievable for a block.
pub fn estimate_compression_ratio<const D: usize>(meta: &BlockMetadata<D>) -> f64 {
    match select_compression(meta) {
        CompressionStrategy::CentroidDelta => COMPACT_COMPRESSION_RATIO,
        CompressionStrategy::Int4Quantize => COMPACT_COMPRESSION_RATIO,
        CompressionStrategy::FullPrecision => NO_COMPRESSION_RATIO,
    }
}

/// Hierarchical tree of geometric block summaries.
///
/// The tree has three fixed levels with a fan-in of four. Level 0 stores the
/// finest blocks, level 1 aggregates four level-0 blocks, and level 2
/// aggregates four level-1 blocks.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HierarchicalBlockTree<const D: usize> {
    #[cfg_attr(
        feature = "serde",
        serde(
            serialize_with = "crate::serde_arrays::serialize_2d",
            deserialize_with = "crate::serde_arrays::deserialize_2d"
        )
    )]
    levels: [[BlockMetadata<D>; MAX_BLOCKS]; 3],
    counts: [usize; 3],
}

impl<const D: usize> HierarchicalBlockTree<D> {
    /// Create an empty hierarchical block tree.
    pub fn new() -> Self {
        Self {
            levels: [[BlockMetadata::empty(); MAX_BLOCKS]; 3],
            counts: [0; 3],
        }
    }

    /// Build the hierarchy from a slice of fine-level blocks.
    pub fn build_from_blocks(&mut self, blocks: &[BlockMetadata<D>]) {
        let n0 = blocks.len().min(MAX_BLOCKS);
        self.levels[0][..n0].copy_from_slice(&blocks[..n0]);
        self.counts[0] = n0;

        let n1 = n0.div_ceil(4);
        for i in 0..n1 {
            let start = i * 4;
            let end = (start + 4).min(n0);
            self.levels[1][i] = aggregate_blocks(&blocks[start..end]);
        }
        self.counts[1] = n1;

        let n2 = n1.div_ceil(4);
        for i in 0..n2 {
            let start = i * 4;
            let end = (start + 4).min(n1);
            self.levels[2][i] = aggregate_blocks(&self.levels[1][start..end]);
        }
        self.counts[2] = n2;
    }

    /// Query the tree and return a boolean mask indicating which level-0
    /// blocks cannot be pruned for `query` given `threshold`.
    pub fn hierarchical_query(&self, query: &[f64; D], threshold: f64) -> [bool; MAX_BLOCKS] {
        let mut result = [false; MAX_BLOCKS];

        let mut active_l2 = [true; MAX_BLOCKS];
        for (i, active) in active_l2.iter_mut().enumerate().take(self.counts[2]) {
            if self.levels[2][i].can_prune(query, threshold) {
                *active = false;
            }
        }

        let mut active_l1 = [false; MAX_BLOCKS];
        for (i, active) in active_l1.iter_mut().enumerate().take(self.counts[1]) {
            let parent = i / 4;
            if parent < self.counts[2]
                && active_l2[parent]
                && !self.levels[1][i].can_prune(query, threshold)
            {
                *active = true;
            }
        }

        for (i, res) in result.iter_mut().enumerate().take(self.counts[0]) {
            let parent = i / 4;
            if parent < self.counts[1]
                && active_l1[parent]
                && !self.levels[0][i].can_prune(query, threshold)
            {
                *res = true;
            }
        }

        result
    }

    /// Return the fraction of level-0 blocks that are marked inactive.
    ///
    /// Returns `0.0` when the tree has no level-0 blocks.
    pub fn pruning_ratio(&self, active_mask: &[bool; MAX_BLOCKS]) -> f64 {
        if self.counts[0] == 0 {
            return 0.0;
        }
        let active = active_mask.iter().filter(|&&x| x).count();
        1.0 - (active as f64 / self.counts[0] as f64)
    }
}

impl<const D: usize> Default for HierarchicalBlockTree<D> {
    fn default() -> Self {
        Self::new()
    }
}

fn aggregate_blocks<const D: usize>(blocks: &[BlockMetadata<D>]) -> BlockMetadata<D> {
    if blocks.is_empty() {
        return BlockMetadata::empty();
    }

    let mut centroid = [0.0f64; D];
    let mut total_count = 0usize;
    let mut max_radius = 0.0f64;
    let mut sum_variance = 0.0f64;
    let mut sum_concentration = 0.0f64;

    for block in blocks {
        let w = block.count as f64;
        for (d, c) in centroid.iter_mut().enumerate() {
            *c += block.centroid[d] * w;
        }
        total_count += block.count;
        max_radius = max_radius.max(block.radius);
        sum_variance += block.variance * w;
        sum_concentration += block.concentration * w;
    }

    if total_count > 0 {
        for c in centroid.iter_mut() {
            *c /= total_count as f64;
        }
    }

    for block in blocks {
        let dist = l2_distance(&centroid, &block.centroid);
        let effective_radius = dist + block.radius;
        max_radius = max_radius.max(effective_radius);
    }

    BlockMetadata {
        centroid,
        radius: max_radius,
        variance: if total_count > 0 {
            sum_variance / total_count as f64
        } else {
            0.0
        },
        concentration: if total_count > 0 {
            sum_concentration / total_count as f64
        } else {
            0.0
        },
        count: total_count,
    }
}

fn l2_distance<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    let mut sum = 0.0;
    for d in 0..D {
        let diff = a[d] - b[d];
        sum += diff * diff;
    }
    sqrt(sum)
}

fn dot_product<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    let mut sum = 0.0;
    for d in 0..D {
        sum += a[d] * b[d];
    }
    sum
}

fn vector_norm<const D: usize>(v: &[f64; D]) -> f64 {
    sqrt(dot_product(v, v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_point_metadata() {
        let points = [[1.0, 2.0, 3.0]];
        let meta = BlockMetadata::from_points(&points);

        assert!((meta.centroid[0] - 1.0).abs() < 1e-10);
        assert_eq!(meta.radius, 0.0);
        assert_eq!(meta.count, 1);
    }

    #[test]
    fn centroid_and_radius() {
        let points = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let meta = BlockMetadata::from_points(&points);

        assert!((meta.centroid[0] - 1.0).abs() < 1e-10);
        assert!((meta.radius - 1.0).abs() < 1e-10);
    }

    #[test]
    fn empty_block_is_empty() {
        let meta = BlockMetadata::<3>::from_points(&[]);
        assert_eq!(meta, BlockMetadata::empty());
    }

    #[test]
    fn pruning_bound() {
        let points = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let meta = BlockMetadata::from_points(&points);
        let query = [1.0, 0.0, 0.0];

        // upper_bound_score = ||q|| * (||centroid|| + radius) = 1 * (1 + 1) = 2.
        assert!(meta.can_prune(&query, 3.0));
        assert!(!meta.can_prune(&query, 1.0));
    }

    #[test]
    fn compression_strategy_selection() {
        let mut meta = BlockMetadata::<3>::empty();

        meta.variance = 0.05;
        assert_eq!(
            select_compression(&meta),
            CompressionStrategy::CentroidDelta
        );

        meta.variance = 0.5;
        meta.concentration = 0.95;
        assert_eq!(select_compression(&meta), CompressionStrategy::Int4Quantize);

        meta.concentration = 0.5;
        assert_eq!(
            select_compression(&meta),
            CompressionStrategy::FullPrecision
        );
    }

    #[test]
    fn hierarchical_query_prunes_blocks() {
        let mut tree = HierarchicalBlockTree::<3>::new();
        let blocks = [
            BlockMetadata::from_points(&[[0.0, 0.0, 0.0]]),
            BlockMetadata::from_points(&[[100.0, 0.0, 0.0]]),
        ];
        tree.build_from_blocks(&blocks);

        // query norm is 1, so block 0 has score 0 and is pruned,
        // while block 1 has score 100 and is kept.
        let query = [1.0, 0.0, 0.0];
        let threshold = 10.0;
        let mask = tree.hierarchical_query(&query, threshold);

        assert!(!mask[0]);
        assert!(mask[1]);
        assert!(tree.pruning_ratio(&mask) > 0.0);
    }

    #[test]
    fn aggregation_counts_children() {
        let blocks = [
            BlockMetadata::from_points(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            BlockMetadata::from_points(&[[4.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
        ];
        let parent = aggregate_blocks(&blocks);

        assert_eq!(parent.count, 4);
        assert!((parent.centroid[0] - 3.0).abs() < 1e-10);
    }
}
