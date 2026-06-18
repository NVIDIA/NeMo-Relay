// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Manifold primitives: points, time-delay embeddings, sparse attention graphs,
//! and geometric concentration.

use libm::sqrt;

/// Default time-delay parameter used when none is supplied.
const DEFAULT_TAU: usize = 1;

/// Maximum number of points stored in a [`SparseAttentionGraph`].
///
/// The limit keeps the adjacency bitmasks within a single `u64` per point
/// and bounds stack usage in `no_std` builds.
const MAX_GRAPH_POINTS: usize = 64;

/// Default capacity of the scalar buffer used by [`TimeDelayEmbedder`].
///
/// The buffer must hold at least `D * tau` samples. A capacity of 256
/// supports embedding dimensions and delays used by the public API.
const EMBED_BUFFER_CAPACITY: usize = 256;

/// Maximum dimension tracked by [`GeometricConcentrator`] for running
/// statistics. Concentration is a dimension-reduction primitive; capping
/// the tracked dimension at 8 keeps the struct small.
const MAX_CONCENTRATOR_DIM: usize = 8;

/// A point in a `D`-dimensional manifold.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManifoldPoint<const D: usize> {
    /// Coordinates of the point.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_arrays"))]
    pub coords: [f64; D],
}

impl<const D: usize> Default for ManifoldPoint<D> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const D: usize> ManifoldPoint<D> {
    /// Create a point at the origin.
    pub const fn zero() -> Self {
        Self { coords: [0.0; D] }
    }

    /// Create a point from its coordinates.
    pub const fn new(coords: [f64; D]) -> Self {
        Self { coords }
    }

    /// Return the Euclidean distance to another point.
    pub fn distance(&self, other: &Self) -> f64 {
        let mut sum = 0.0;
        for i in 0..D {
            let d = self.coords[i] - other.coords[i];
            sum += d * d;
        }
        sqrt(sum)
    }

    /// Return true if this point lies within `epsilon` of `other`.
    pub fn is_neighbor(&self, other: &Self, epsilon: f64) -> bool {
        self.distance(other) < epsilon
    }
}

/// Time-delay embedder that maps a 1-D scalar signal into a `D`-dimensional
/// manifold according to Takens' embedding theorem.
///
/// The embedding at time `t` is `[x(t), x(t-tau), x(t-2*tau), ...]`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeDelayEmbedder<const D: usize> {
    tau: usize,
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_arrays"))]
    buffer: [f64; EMBED_BUFFER_CAPACITY],
    buffer_pos: usize,
    buffer_len: usize,
}

impl<const D: usize> TimeDelayEmbedder<D> {
    /// Create an embedder with the supplied delay. A zero delay is normalized
    /// to `DEFAULT_TAU`.
    ///
    /// # Panics
    ///
    /// Panics if `D * tau` exceeds `EMBED_BUFFER_CAPACITY`. This is a
    /// programming error, not a runtime failure.
    pub fn new(tau: usize) -> Self {
        let tau = if tau == 0 { DEFAULT_TAU } else { tau };
        assert!(
            D * tau <= EMBED_BUFFER_CAPACITY,
            "TimeDelayEmbedder D * tau ({}) exceeds EMBED_BUFFER_CAPACITY ({})",
            D * tau,
            EMBED_BUFFER_CAPACITY
        );
        Self {
            tau,
            buffer: [0.0; EMBED_BUFFER_CAPACITY],
            buffer_pos: 0,
            buffer_len: 0,
        }
    }

    /// Push a new scalar sample into the buffer.
    pub fn push(&mut self, value: f64) {
        self.buffer[self.buffer_pos] = value;
        self.buffer_pos = (self.buffer_pos + 1) % EMBED_BUFFER_CAPACITY;
        if self.buffer_len < EMBED_BUFFER_CAPACITY {
            self.buffer_len += 1;
        }
    }

    /// Return the current embedded point, or `None` if the buffer does not
    /// yet contain enough samples.
    pub fn embed(&self) -> Option<ManifoldPoint<D>> {
        let required = D * self.tau;
        if self.buffer_len < required {
            return None;
        }

        let mut point = ManifoldPoint::zero();
        for i in 0..D {
            let offset = i * self.tau;
            let idx =
                (self.buffer_pos + EMBED_BUFFER_CAPACITY - 1 - offset) % EMBED_BUFFER_CAPACITY;
            point.coords[i] = self.buffer[idx];
        }

        Some(point)
    }

    /// Reset the embedder to an empty state.
    pub fn reset(&mut self) {
        self.buffer = [0.0; EMBED_BUFFER_CAPACITY];
        self.buffer_pos = 0;
        self.buffer_len = 0;
    }
}

/// Sparse attention graph where edges exist only between points within an
/// epsilon-neighborhood.
///
/// The adjacency is stored as a `u64` bitmask per point, limiting the graph
/// to `MAX_GRAPH_POINTS` vertices.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseAttentionGraph<const D: usize> {
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_arrays"))]
    points: [ManifoldPoint<D>; MAX_GRAPH_POINTS],
    point_count: usize,
    epsilon: f64,
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_arrays"))]
    adjacency: [u64; MAX_GRAPH_POINTS],
}

impl<const D: usize> SparseAttentionGraph<D> {
    /// Create an empty graph with the given neighborhood radius.
    pub fn new(epsilon: f64) -> Self {
        Self {
            points: [ManifoldPoint::zero(); MAX_GRAPH_POINTS],
            point_count: 0,
            epsilon,
            adjacency: [0; MAX_GRAPH_POINTS],
        }
    }

    /// Add a point to the graph and return its index, or `None` if the
    /// graph is full.
    pub fn add_point(&mut self, point: ManifoldPoint<D>) -> Option<usize> {
        if self.point_count >= MAX_GRAPH_POINTS {
            return None;
        }

        let idx = self.point_count;
        self.points[idx] = point;

        let mut mask = 0u64;
        for i in 0..idx {
            if point.is_neighbor(&self.points[i], self.epsilon) {
                mask |= 1 << i;
                self.adjacency[i] |= 1 << idx;
            }
        }
        self.adjacency[idx] = mask;

        self.point_count += 1;
        Some(idx)
    }

    /// Return the number of neighbors (degree) of point `idx`.
    pub fn degree(&self, idx: usize) -> u32 {
        if idx >= self.point_count {
            return 0;
        }
        self.adjacency[idx].count_ones()
    }

    /// Return true if points `i` and `j` are neighbors.
    pub fn are_neighbors(&self, i: usize, j: usize) -> bool {
        if i >= self.point_count || j >= self.point_count {
            return false;
        }
        (self.adjacency[i] & (1 << j)) != 0
    }

    /// Count connected components (`β₀`) using iterative depth-first search.
    pub fn compute_betti_0(&self) -> u32 {
        if self.point_count == 0 {
            return 0;
        }

        let mut visited = [false; MAX_GRAPH_POINTS];
        let mut components = 0u32;

        for start in 0..self.point_count {
            if visited[start] {
                continue;
            }

            components += 1;
            let mut stack = [0usize; MAX_GRAPH_POINTS];
            let mut stack_top = 1;
            stack[0] = start;

            while stack_top > 0 {
                stack_top -= 1;
                let current = stack[stack_top];

                if visited[current] {
                    continue;
                }
                visited[current] = true;

                for (neighbor, is_visited) in visited.iter().enumerate().take(self.point_count) {
                    if !is_visited
                        && self.are_neighbors(current, neighbor)
                        && stack_top < MAX_GRAPH_POINTS
                    {
                        stack[stack_top] = neighbor;
                        stack_top += 1;
                    }
                }
            }
        }

        components
    }

    /// Estimate the number of 1-dimensional holes (`β₁`) using the Euler
    /// characteristic approximation `β₁ ≈ E - V + β₀`.
    pub fn estimate_betti_1(&self) -> u32 {
        let v = self.point_count as i32;
        let mut e = 0i32;

        for i in 0..self.point_count {
            e += self.adjacency[i].count_ones() as i32;
        }
        e /= 2;

        let b0 = self.compute_betti_0() as i32;
        let b1 = e - v + b0;
        b1.max(0) as u32
    }

    /// Return the topological shape signature `(β₀, β₁)`.
    pub fn shape(&self) -> (u32, u32) {
        (self.compute_betti_0(), self.estimate_betti_1())
    }

    /// Clear all points and edges from the graph.
    pub fn clear(&mut self) {
        self.point_count = 0;
        self.adjacency = [0; MAX_GRAPH_POINTS];
    }
}

/// Streaming geometric concentrator that tracks per-coordinate mean and
/// variance to identify the principal axis of a point cloud.
///
/// Only the first eight coordinates of each point participate in the
/// running statistics; dimensions beyond 8 are ignored.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometricConcentrator<const D: usize> {
    mean: [f64; MAX_CONCENTRATOR_DIM],
    variance: [f64; MAX_CONCENTRATOR_DIM],
    count: u64,
}

impl<const D: usize> GeometricConcentrator<D> {
    /// Create a new concentrator with zero statistics.
    pub fn new() -> Self {
        Self {
            mean: [0.0; MAX_CONCENTRATOR_DIM],
            variance: [0.0; MAX_CONCENTRATOR_DIM],
            count: 0,
        }
    }

    /// Update running statistics with a new point using Welford's algorithm.
    pub fn update(&mut self, point: &ManifoldPoint<D>) {
        self.count += 1;
        let n = self.count as f64;

        for i in 0..D.min(MAX_CONCENTRATOR_DIM) {
            let delta = point.coords[i] - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = point.coords[i] - self.mean[i];
            self.variance[i] += delta * delta2;
        }
    }

    /// Return the index of the dimension with the largest variance.
    pub fn principal_dimension(&self) -> usize {
        let mut max_var = 0.0;
        let mut max_dim = 0;

        if self.count > 0 {
            for i in 0..D.min(MAX_CONCENTRATOR_DIM) {
                let var = self.variance[i] / self.count as f64;
                if var > max_var {
                    max_var = var;
                    max_dim = i;
                }
            }
        }

        max_dim
    }

    /// Project a point onto the principal axis relative to the running mean.
    pub fn concentrate_1d(&self, point: &ManifoldPoint<D>) -> f64 {
        let dim = self.principal_dimension();
        if dim < D {
            point.coords[dim] - self.mean[dim]
        } else {
            0.0
        }
    }

    /// Return the fraction of total variance captured by the principal
    /// dimension.
    pub fn concentration_ratio(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }

        let dims = D.min(MAX_CONCENTRATOR_DIM);
        let total: f64 = self.variance[..dims].iter().sum::<f64>() / self.count as f64;
        if total == 0.0 {
            return 0.0;
        }

        let principal = self.variance[self.principal_dimension()] / self.count as f64;
        principal / total
    }

    /// Reset the concentrator to its initial state.
    pub fn reset(&mut self) {
        self.mean = [0.0; MAX_CONCENTRATOR_DIM];
        self.variance = [0.0; MAX_CONCENTRATOR_DIM];
        self.count = 0;
    }
}

impl<const D: usize> Default for GeometricConcentrator<D> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_distance() {
        let p1 = ManifoldPoint::<3>::new([0.0, 0.0, 0.0]);
        let p2 = ManifoldPoint::<3>::new([3.0, 4.0, 0.0]);
        assert!((p1.distance(&p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn neighbor_check_respects_epsilon() {
        let p1 = ManifoldPoint::<3>::new([0.0, 0.0, 0.0]);
        let p2 = ManifoldPoint::<3>::new([0.1, 0.1, 0.0]);

        assert!(p1.is_neighbor(&p2, 0.5));
        assert!(!p1.is_neighbor(&p2, 0.1));
    }

    #[test]
    fn time_delay_embedding() {
        let mut embedder = TimeDelayEmbedder::<3>::new(1);
        for i in 0..10 {
            embedder.push(i as f64);
        }

        let point = embedder.embed().unwrap();
        assert!((point.coords[0] - 9.0).abs() < 1e-10);
        assert!((point.coords[1] - 8.0).abs() < 1e-10);
        assert!((point.coords[2] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn embedder_returns_none_until_ready() {
        let mut embedder = TimeDelayEmbedder::<3>::new(2);
        for i in 0..5 {
            embedder.push(i as f64);
        }
        assert!(embedder.embed().is_none());

        embedder.push(5.0);
        assert!(embedder.embed().is_some());
    }

    #[test]
    #[should_panic(expected = "exceeds EMBED_BUFFER_CAPACITY")]
    fn embedder_panics_for_excessive_delay() {
        let _ = TimeDelayEmbedder::<256>::new(2);
    }

    #[test]
    fn single_component_graph() {
        let mut graph = SparseAttentionGraph::<3>::new(1.0);
        graph.add_point(ManifoldPoint::new([0.0, 0.0, 0.0]));
        graph.add_point(ManifoldPoint::new([0.5, 0.0, 0.0]));
        graph.add_point(ManifoldPoint::new([0.5, 0.5, 0.0]));

        assert_eq!(graph.compute_betti_0(), 1);
    }

    #[test]
    fn disconnected_components() {
        let mut graph = SparseAttentionGraph::<2>::new(0.1);
        graph.add_point(ManifoldPoint::new([0.0, 0.0]));
        graph.add_point(ManifoldPoint::new([10.0, 0.0]));
        graph.add_point(ManifoldPoint::new([10.1, 0.0]));

        assert_eq!(graph.compute_betti_0(), 2);
    }

    #[test]
    fn graph_full_returns_none() {
        let mut graph = SparseAttentionGraph::<1>::new(1.0);
        for _ in 0..MAX_GRAPH_POINTS {
            assert!(graph.add_point(ManifoldPoint::new([0.0])).is_some());
        }
        assert!(graph.add_point(ManifoldPoint::new([0.0])).is_none());
    }

    #[test]
    fn concentrator_principal_axis() {
        let mut concentrator = GeometricConcentrator::<3>::new();
        concentrator.update(&ManifoldPoint::new([10.0, 0.0, 0.0]));
        concentrator.update(&ManifoldPoint::new([11.0, 0.0, 0.0]));
        concentrator.update(&ManifoldPoint::new([12.0, 0.0, 0.0]));

        assert_eq!(concentrator.principal_dimension(), 0);
        assert!(concentrator.concentration_ratio() > 0.99);
    }
}
