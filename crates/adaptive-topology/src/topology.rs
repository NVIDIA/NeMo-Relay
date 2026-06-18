// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Topological shape signatures and binary-data verification primitives.

use libm::sqrt;

/// Distance threshold used to separate connected components when computing
/// `β₀` over a 1-D byte stream.
const CLUSTER_THRESHOLD: i16 = 15;

/// Sliding window size used for localized topology analysis.
const WINDOW_SIZE: usize = 64;

/// Minimum component density (`β₀ / len`) expected for valid data.
const DENSITY_MIN: f64 = 0.1;

/// Maximum component density expected for valid data.
const DENSITY_MAX: f64 = 0.6;

/// Maximum allowed `β₁` loop count per window.
const MAX_BETTI_1: u32 = 10;

/// Tolerance used to decide whether two byte values "close" a loop when
/// computing `β₁`.
const LOOP_TOLERANCE: i16 = 5;

/// Topological shape signature of a byte sequence.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopologicalShape {
    /// Number of connected components (`β₀`).
    pub betti_0: u32,
    /// Heuristic estimate of the number of 1-dimensional holes (loops).
    pub betti_1: u32,
    /// Normalized component density (`β₀ / data_length`).
    pub density: f64,
}

impl TopologicalShape {
    /// Create a shape from Betti numbers and the source data length.
    pub fn new(betti_0: u32, betti_1: u32, data_len: usize) -> Self {
        let density = if data_len > 0 {
            betti_0 as f64 / data_len as f64
        } else {
            0.0
        };

        Self {
            betti_0,
            betti_1,
            density,
        }
    }

    /// Return the Euclidean distance between two shape signatures in
    /// `(β₀, β₁, density)` space.
    pub fn distance(&self, other: &Self) -> f64 {
        let d0 = self.betti_0 as f64 - other.betti_0 as f64;
        let d1 = self.betti_1 as f64 - other.betti_1 as f64;
        let dd = self.density - other.density;

        sqrt(d0 * d0 + d1 * d1 + dd * dd)
    }
}

/// Approximate `β₀` for a 1-D byte stream by counting gaps larger than
/// `CLUSTER_THRESHOLD`.
pub fn compute_betti_0(data: &[u8]) -> u32 {
    if data.is_empty() {
        return 0;
    }

    let mut components = 1u32;

    for window in data.windows(2) {
        let dist = (window[0] as i16 - window[1] as i16).abs();

        if dist > CLUSTER_THRESHOLD {
            components += 1;
        }
    }

    components
}

/// Heuristic estimate of loop count for a 1-D byte stream.
///
/// This detects short return patterns of the form `a -> b -> c -> ~a`. It is
/// not a true topological `β₁` computation; it is a fast, deterministic
/// approximation used by [`compute_shape`].
pub fn estimate_loop_count(data: &[u8]) -> u32 {
    if data.len() < 4 {
        return 0;
    }

    let mut loops = 0u32;

    for window in data.windows(4) {
        let a = window[0] as i16;
        let b = window[1] as i16;
        let c = window[2] as i16;
        let d = window[3] as i16;

        if (a - d).abs() <= LOOP_TOLERANCE {
            // Count only loops where the middle values actually traverse
            // away from the start value.
            if (a - b).abs() > LOOP_TOLERANCE || (a - c).abs() > LOOP_TOLERANCE {
                loops += 1;
            }
        }
    }

    loops
}

/// Compute the full topological shape signature of a byte sequence.
pub fn compute_shape(data: &[u8]) -> TopologicalShape {
    let betti_0 = compute_betti_0(data);
    let betti_1 = estimate_loop_count(data);
    TopologicalShape::new(betti_0, betti_1, data.len())
}

/// Result of verifying a byte sequence against topological constraints.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VerifyResult {
    /// The data passed all topological checks.
    Pass,
    /// The component density fell outside the expected range.
    InvalidDensity {
        /// Observed density.
        actual: f64,
        /// Minimum allowed density.
        min: f64,
        /// Maximum allowed density.
        max: f64,
    },
    /// The loop count exceeded the allowed maximum.
    ExcessiveLoops {
        /// Observed loop count.
        count: u32,
        /// Maximum allowed loop count.
        max: u32,
    },
    /// The shape differed from the reference by more than the threshold.
    ShapeMismatch {
        /// Observed distance to the reference.
        distance: f64,
        /// Allowed distance threshold.
        threshold: f64,
    },
}

/// Verify a byte sequence against built-in density and loop-complexity
/// heuristics.
pub fn verify_shape(data: &[u8]) -> VerifyResult {
    let shape = compute_shape(data);

    if shape.density < DENSITY_MIN || shape.density > DENSITY_MAX {
        return VerifyResult::InvalidDensity {
            actual: shape.density,
            min: DENSITY_MIN,
            max: DENSITY_MAX,
        };
    }

    if shape.betti_1 > MAX_BETTI_1 {
        return VerifyResult::ExcessiveLoops {
            count: shape.betti_1,
            max: MAX_BETTI_1,
        };
    }

    VerifyResult::Pass
}

/// Convenience wrapper that returns `true` if `verify_shape` passes.
pub fn is_shape_valid(data: &[u8]) -> bool {
    matches!(verify_shape(data), VerifyResult::Pass)
}

/// Verify a byte sequence against a reference shape using a Wasserstein-like
/// distance, then apply the standard shape checks.
pub fn verify_against_reference(
    data: &[u8],
    reference: &TopologicalShape,
    threshold: f64,
) -> VerifyResult {
    let shape = compute_shape(data);
    let distance = shape.distance(reference);

    if distance > threshold {
        return VerifyResult::ShapeMismatch {
            distance,
            threshold,
        };
    }

    verify_shape(data)
}

/// Apply `is_shape_valid` over a sliding window of size `window_size`.
///
/// A `window_size` of zero defaults to `WINDOW_SIZE`. Returns `Ok(())` if
/// every window passes, or `Err(offset)` at the first failing offset.
pub fn verify_sliding_window(data: &[u8], window_size: usize) -> Result<(), usize> {
    let size = if window_size == 0 {
        WINDOW_SIZE
    } else {
        window_size
    };

    if data.len() < size {
        return if is_shape_valid(data) { Ok(()) } else { Err(0) };
    }

    for (offset, window) in data.windows(size).enumerate() {
        if !is_shape_valid(window) {
            return Err(offset);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_data_has_zero_shape() {
        assert_eq!(compute_betti_0(&[]), 0);
        assert_eq!(estimate_loop_count(&[]), 0);
        let shape = compute_shape(&[]);
        assert_eq!(shape.density, 0.0);
    }

    #[test]
    fn single_component_for_uniform_data() {
        let nop_sled = [0x90u8; 64];
        let shape = compute_shape(&nop_sled);
        assert_eq!(shape.betti_0, 1);
    }

    #[test]
    fn two_components_for_single_gap() {
        assert_eq!(compute_betti_0(&[0u8, 100]), 2);
    }

    #[test]
    fn shape_distance() {
        let a = TopologicalShape::new(1, 0, 10);
        let b = TopologicalShape::new(2, 1, 10);
        assert!(a.distance(&b) > 0.0);
        assert_eq!(a.distance(&a), 0.0);
    }

    #[test]
    fn verify_shape_returns_reason() {
        // A 64-byte uniform block has density 0, which fails density check.
        let data = [0x90u8; 64];
        let result = verify_shape(&data);
        assert!(matches!(result, VerifyResult::InvalidDensity { .. }));
    }

    #[test]
    fn reference_verification_rejects_distant_shape() {
        let data = [0x00u8; 16];
        let reference = TopologicalShape::new(10, 5, 16);
        let result = verify_against_reference(&data, &reference, 1.0);
        assert!(matches!(result, VerifyResult::ShapeMismatch { .. }));
    }

    #[test]
    fn sliding_window_short_input() {
        // A single-byte input has density 1.0, which fails the density check.
        let data = [0x90u8];
        assert_eq!(verify_sliding_window(&data, 0), Err(0));
    }

    #[test]
    fn sliding_window_passes_multiple_windows() {
        // 128 bytes of grouped values: every 64-byte window has 8 components
        // and zero loops, giving a valid density of 8/64.
        let mut data = [0u8; 128];
        for (i, byte) in data.iter_mut().enumerate() {
            let group = i / 8;
            *byte = if group % 2 == 0 { 0 } else { 100 };
        }
        assert!(verify_sliding_window(&data, 64).is_ok());
    }

    #[test]
    fn sliding_window_first_passes_later_fails() {
        // First 64 bytes are valid; the remaining 64 bytes are uniform and
        // fail the density check, causing a later window to error.
        let mut data = [0x90u8; 128];
        for (i, byte) in data.iter_mut().enumerate().take(64) {
            let group = i / 8;
            *byte = if group % 2 == 0 { 0 } else { 100 };
        }
        let result = verify_sliding_window(&data, 64);
        assert!(result.is_err());
        assert!(result.unwrap_err() > 0);
    }

    #[test]
    fn sliding_window_boundary_full_window() {
        // When the input length exactly equals the window size, one window
        // is verified and the call succeeds for valid data.
        let mut data = [0u8; 64];
        for (i, byte) in data.iter_mut().enumerate() {
            let group = i / 8;
            *byte = if group % 2 == 0 { 0 } else { 100 };
        }
        assert!(verify_sliding_window(&data, 64).is_ok());
    }
}
