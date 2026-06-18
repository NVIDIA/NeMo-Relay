// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-facing bindings for adaptive topology primitives.

use nemo_relay_adaptive_topology::{
    BettiNumbers, ConvergenceDetector, DriftDetector, GeometricGovernor,
};
use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Python wrapper for [`GeometricGovernor`].
#[pyclass(name = "GeometricGovernor")]
pub struct PyGeometricGovernor {
    inner: GeometricGovernor,
}

#[pymethods]
impl PyGeometricGovernor {
    /// Create a new governor with the default threshold and gains.
    #[new]
    #[pyo3(text_signature = "()")]
    fn new() -> Self {
        Self {
            inner: GeometricGovernor::new(),
        }
    }

    /// Create a governor with a custom initial threshold.
    #[staticmethod]
    #[pyo3(text_signature = "(epsilon)")]
    fn with_epsilon(epsilon: f64) -> Self {
        Self {
            inner: GeometricGovernor::with_epsilon(epsilon),
        }
    }

    /// Adapt epsilon based on the observed deviation over `dt` seconds.
    #[pyo3(text_signature = "(deviation_delta, dt)")]
    fn adapt(&mut self, deviation_delta: f64, dt: f64) -> f64 {
        self.inner.adapt(deviation_delta, dt)
    }

    /// Return true if `deviation` meets or exceeds the current threshold.
    #[pyo3(text_signature = "(deviation)")]
    fn should_trigger(&self, deviation: f64) -> bool {
        self.inner.should_trigger(deviation)
    }

    /// Return the current epsilon value.
    #[getter]
    fn epsilon(&self) -> f64 {
        self.inner.epsilon()
    }

    /// Return the number of adaptations performed so far.
    #[getter]
    fn adjustment_count(&self) -> u64 {
        self.inner.adjustment_count()
    }

    /// Return the error from the most recent adaptation.
    #[getter]
    fn last_error(&self) -> f64 {
        self.inner.last_error()
    }

    /// Reset the governor to its initial state.
    #[pyo3(text_signature = "()")]
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Return a debug representation of the governor.
    fn __repr__(&self) -> String {
        format!(
            "GeometricGovernor(epsilon={}, adjustment_count={})",
            self.inner.epsilon(),
            self.inner.adjustment_count()
        )
    }
}

/// Python wrapper for [`ConvergenceDetector`].
#[pyclass(name = "ConvergenceDetector")]
pub struct PyConvergenceDetector {
    inner: ConvergenceDetector,
}

#[pymethods]
impl PyConvergenceDetector {
    /// Create a new detector.
    ///
    /// `epsilon` is the error threshold. `stability_window` is the minimum
    /// number of epochs required to judge stability.
    #[new]
    #[pyo3(text_signature = "(epsilon, stability_window)")]
    fn new(epsilon: f64, stability_window: usize) -> Self {
        Self {
            inner: ConvergenceDetector::new(epsilon, stability_window),
        }
    }

    /// Record an epoch's metrics.
    #[pyo3(text_signature = "(beta_0, beta_1, drift, error)")]
    fn record_epoch(&mut self, beta_0: u32, beta_1: u32, drift: f64, error: f64) {
        self.inner
            .record_epoch(BettiNumbers::new(beta_0, beta_1), drift, error);
    }

    /// Return true if convergence criteria are satisfied.
    #[getter]
    fn is_converged(&self) -> bool {
        self.inner.is_converged()
    }

    /// Return a score in `[0, 1]` indicating how close to converged the
    /// detector is, where `1.0` means fully converged.
    #[getter]
    fn convergence_score(&self) -> f64 {
        self.inner.convergence_score()
    }

    /// Return the number of epochs recorded.
    #[getter]
    fn epoch(&self) -> u32 {
        self.inner.epoch()
    }

    /// Return the most recent Betti numbers as ``(beta_0, beta_1)``, or ``None``.
    #[getter]
    fn last_betti(&self) -> Option<(u32, u32)> {
        self.inner.last_betti().map(|b| (b.beta_0, b.beta_1))
    }

    /// Return the most recent error, or ``None``.
    #[getter]
    fn last_error(&self) -> Option<f64> {
        self.inner.last_error()
    }

    /// Reset the detector to its initial state.
    #[pyo3(text_signature = "()")]
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Return a debug representation of the detector.
    fn __repr__(&self) -> String {
        format!(
            "ConvergenceDetector(epoch={}, is_converged={})",
            self.inner.epoch(),
            if self.inner.is_converged() {
                "True"
            } else {
                "False"
            }
        )
    }
}

/// Python wrapper for a 3-dimensional [`DriftDetector`].
#[pyclass(name = "DriftDetector")]
pub struct PyDriftDetector {
    inner: DriftDetector<3>,
}

#[pymethods]
impl PyDriftDetector {
    /// Create a new detector with no previous observation and zero velocity.
    #[new]
    #[pyo3(text_signature = "()")]
    fn new() -> Self {
        Self {
            inner: DriftDetector::new(),
        }
    }

    /// Record a new centroid and return the drift from the predicted position.
    #[pyo3(text_signature = "(centroid)")]
    fn update(&mut self, centroid: [f64; 3]) -> f64 {
        self.inner.update(&centroid)
    }

    /// Return true if the current velocity magnitude exceeds `threshold`.
    #[pyo3(text_signature = "(threshold)")]
    fn is_drifting(&self, threshold: f64) -> bool {
        self.inner.is_drifting(threshold)
    }

    /// Return the magnitude of the current velocity vector.
    #[pyo3(text_signature = "()")]
    fn velocity_magnitude(&self) -> f64 {
        self.inner.velocity_magnitude()
    }

    /// Reset the detector to its initial state.
    #[pyo3(text_signature = "()")]
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Return a debug representation of the detector.
    fn __repr__(&self) -> String {
        format!(
            "DriftDetector(velocity_magnitude={})",
            self.inner.velocity_magnitude()
        )
    }
}

/// Register the adaptive topology classes in the `_native` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGeometricGovernor>()?;
    m.add_class::<PyConvergenceDetector>()?;
    m.add_class::<PyDriftDetector>()?;
    Ok(())
}
