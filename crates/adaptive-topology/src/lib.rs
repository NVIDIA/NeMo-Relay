// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Topology, manifold, drift, and adaptive-threshold primitives for NeMo Relay.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "serde")]
pub(crate) mod serde_arrays {
    //! Helpers for serializing fixed-size arrays that serde does not handle
    //! natively (const-generic sizes or lengths above the built-in limit).

    use core::fmt;
    use core::marker::PhantomData;
    use serde::de::{SeqAccess, Visitor};
    use serde::ser::SerializeTuple;
    use serde::{Deserializer, Serializer};

    /// Serialize a fixed-size array as a tuple sequence.
    pub fn serialize<T, S, const N: usize>(value: &[T; N], serializer: S) -> Result<S::Ok, S::Error>
    where
        T: serde::Serialize,
        S: Serializer,
    {
        let mut tuple = serializer.serialize_tuple(N)?;
        for item in value.iter() {
            tuple.serialize_element(item)?;
        }
        tuple.end()
    }

    /// Deserialize a fixed-size array from a tuple sequence.
    pub fn deserialize<'de, T, D, const N: usize>(deserializer: D) -> Result<[T; N], D::Error>
    where
        T: serde::Deserialize<'de> + Default + Copy,
        D: Deserializer<'de>,
    {
        struct ArrayVisitor<T, const N: usize>(PhantomData<T>);

        impl<'de, T, const N: usize> Visitor<'de> for ArrayVisitor<T, N>
        where
            T: serde::Deserialize<'de> + Default + Copy,
        {
            type Value = [T; N];

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                write!(formatter, "an array of {} elements", N)
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut arr = [T::default(); N];
                for (i, slot) in arr.iter_mut().enumerate() {
                    *slot = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
                }
                Ok(arr)
            }
        }

        deserializer.deserialize_tuple(N, ArrayVisitor(PhantomData))
    }

    /// Serialize a fixed-size two-dimensional array as a flat tuple sequence.
    pub fn serialize_2d<T, S, const M: usize, const N: usize>(
        value: &[[T; N]; M],
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        T: serde::Serialize,
        S: Serializer,
    {
        let mut tuple = serializer.serialize_tuple(M * N)?;
        for row in value.iter() {
            for item in row.iter() {
                tuple.serialize_element(item)?;
            }
        }
        tuple.end()
    }

    /// Deserialize a fixed-size two-dimensional array from a flat tuple sequence.
    pub fn deserialize_2d<'de, T, D, const M: usize, const N: usize>(
        deserializer: D,
    ) -> Result<[[T; N]; M], D::Error>
    where
        T: serde::Deserialize<'de> + Default + Copy,
        D: Deserializer<'de>,
    {
        struct Array2Visitor<T, const M: usize, const N: usize>(PhantomData<T>);

        impl<'de, T, const M: usize, const N: usize> Visitor<'de> for Array2Visitor<T, M, N>
        where
            T: serde::Deserialize<'de> + Default + Copy,
        {
            type Value = [[T; N]; M];

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                write!(formatter, "a {}x{} array", M, N)
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut arr = [[T::default(); N]; M];
                for (i, row) in arr.iter_mut().enumerate() {
                    for (j, slot) in row.iter_mut().enumerate() {
                        *slot = seq
                            .next_element()?
                            .ok_or_else(|| serde::de::Error::invalid_length(i * N + j, &self))?;
                    }
                }
                Ok(arr)
            }
        }

        deserializer.deserialize_tuple(M * N, Array2Visitor(PhantomData))
    }
}

pub mod convergence;
pub mod drift;
pub mod geometry;
pub mod governor;
pub mod manifold;
pub mod topology;

pub use convergence::{BettiNumbers, ConvergenceDetector};
pub use drift::DriftDetector;
pub use governor::GeometricGovernor;

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;
    use geometry::{BlockMetadata, CompressionStrategy, HierarchicalBlockTree};
    use manifold::{GeometricConcentrator, ManifoldPoint, SparseAttentionGraph, TimeDelayEmbedder};
    use topology::{TopologicalShape, VerifyResult};

    #[test]
    fn round_trip_public_types() {
        let governor = GeometricGovernor::new();
        let governor_json = serde_json::to_string(&governor).unwrap();
        let governor_back: GeometricGovernor = serde_json::from_str(&governor_json).unwrap();
        assert_eq!(governor, governor_back);

        let betti = BettiNumbers::new(1, 0);
        let betti_json = serde_json::to_string(&betti).unwrap();
        let betti_back: BettiNumbers = serde_json::from_str(&betti_json).unwrap();
        assert_eq!(betti, betti_back);

        let shape = TopologicalShape::new(2, 1, 100);
        let shape_json = serde_json::to_string(&shape).unwrap();
        let shape_back: TopologicalShape = serde_json::from_str(&shape_json).unwrap();
        assert_eq!(shape, shape_back);

        let verify = VerifyResult::Pass;
        let verify_json = serde_json::to_string(&verify).unwrap();
        let verify_back: VerifyResult = serde_json::from_str(&verify_json).unwrap();
        assert_eq!(verify, verify_back);

        let point = ManifoldPoint::<3>::new([1.0, 2.0, 3.0]);
        let point_json = serde_json::to_string(&point).unwrap();
        let point_back: ManifoldPoint<3> = serde_json::from_str(&point_json).unwrap();
        assert_eq!(point, point_back);

        let mut embedder = TimeDelayEmbedder::<2>::new(1);
        embedder.push(1.0);
        embedder.push(2.0);
        embedder.push(3.0);
        let embedder_json = serde_json::to_string(&embedder).unwrap();
        let embedder_back: TimeDelayEmbedder<2> = serde_json::from_str(&embedder_json).unwrap();
        assert_eq!(embedder, embedder_back);

        let mut graph = SparseAttentionGraph::<3>::new(1.0);
        graph.add_point(ManifoldPoint::new([0.0, 0.0, 0.0]));
        graph.add_point(ManifoldPoint::new([0.5, 0.0, 0.0]));
        let graph_json = serde_json::to_string(&graph).unwrap();
        let graph_back: SparseAttentionGraph<3> = serde_json::from_str(&graph_json).unwrap();
        assert_eq!(graph, graph_back);

        let mut concentrator = GeometricConcentrator::<3>::new();
        concentrator.update(&ManifoldPoint::new([1.0, 0.0, 0.0]));
        let concentrator_json = serde_json::to_string(&concentrator).unwrap();
        let concentrator_back: GeometricConcentrator<3> =
            serde_json::from_str(&concentrator_json).unwrap();
        assert_eq!(concentrator, concentrator_back);

        let block = BlockMetadata::<3>::from_points(&[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        let block_json = serde_json::to_string(&block).unwrap();
        let block_back: BlockMetadata<3> = serde_json::from_str(&block_json).unwrap();
        assert_eq!(block, block_back);

        let strategy = CompressionStrategy::Int4Quantize;
        let strategy_json = serde_json::to_string(&strategy).unwrap();
        let strategy_back: CompressionStrategy = serde_json::from_str(&strategy_json).unwrap();
        assert_eq!(strategy, strategy_back);

        let mut tree = HierarchicalBlockTree::<3>::new();
        tree.build_from_blocks(&[block]);
        let tree_json = serde_json::to_string(&tree).unwrap();
        let tree_back: HierarchicalBlockTree<3> = serde_json::from_str(&tree_json).unwrap();
        assert_eq!(tree, tree_back);

        let mut drift = DriftDetector::<3>::new();
        drift.update(&[1.0, 0.0, 0.0]);
        drift.update(&[2.0, 0.0, 0.0]);
        let drift_json = serde_json::to_string(&drift).unwrap();
        let drift_back: DriftDetector<3> = serde_json::from_str(&drift_json).unwrap();
        assert_eq!(drift, drift_back);

        let mut detector = ConvergenceDetector::new(0.001, 3);
        detector.record_epoch(BettiNumbers::new(1, 0), 0.05, 0.04);
        detector.record_epoch(BettiNumbers::new(1, 0), 0.001, 0.02);
        let detector_json = serde_json::to_string(&detector).unwrap();
        let detector_back: ConvergenceDetector = serde_json::from_str(&detector_json).unwrap();
        assert_eq!(detector, detector_back);
    }
}
