//! Provides the basic types common to most optimizers.

use std::ops::Deref;


/// Defines an objective function `f` that is subject to minimization.
pub trait Function {
    /// Computes the objective function at a given `position` `x`, i.e., `f(x) = y`.
    fn value(&self, position: &[f64]) -> f64;
}

impl<'a, F: 'a + Function> Function for &'a F {
    fn value(&self, position: &[f64]) -> f64 {
        (*self).value(position)
    }
}


/// Defines an analytical differentiable objective function.
///
/// In extension to `Objective`, this objective function is able to compute the gradient
/// at a given position.
pub trait DifferentiableFunction: Function {
    /// Computes the gradient of the objective function at a given `position` `x`,
    /// i.e., `∀ᵢ ∂/∂xᵢ f(x)`.
    fn gradient(&self, position: &[f64]) -> Vec<f64>;

    fn probe(&self, position: &[f64]) -> (f64, Vec<f64>) {
        (self.value(position), self.gradient(position))
    }
}

impl<'a, F: 'a + DifferentiableFunction> DifferentiableFunction for &'a F {
    fn gradient(&self, position: &[f64]) -> Vec<f64> {
        (*self).gradient(position)
    }
}


/// Represents a summation of individual objective functions.
///
/// The combination is represented as a sum of the individual functions, i.e.,
/// `f(x) = ∑ᵢ fᵢ(x)`. Same applies to the gradient, obviously.
///
/// Some optimizers, e.g., `StochasticGradientDescent`, exploit this fact.
pub struct Summation<C> {
    terms: C
}

impl<C: Deref<Target=[F]>, F: Function> Summation<C> {
    /// Creates a new summation given the terms, i.e., individual functions to sum up.
    pub fn new(terms: C) -> Self {
        Summation {
            terms: terms
        }
    }

    /// Returns the functions that are summed up.
    pub fn terms(&self) -> &[F] {
        &*self.terms
    }
}

impl<C: Deref<Target=[F]>, F: Function> Function for Summation<C> {
    fn value(&self, position: &[f64]) -> f64 {
        let mut value = 0.0;

        for term in self.terms() {
            value += term.value(position);
        }

        value
    }
}

impl<C: Deref<Target=[F]>, F: DifferentiableFunction> DifferentiableFunction for Summation<C> {
    fn gradient(&self, position: &[f64]) -> Vec<f64> {
        let mut gradient = vec![0.0; position.len()];

        // TODO: This can be optimized easily
        for term in self.terms() {
            for (g, gi) in gradient.iter_mut().zip(term.gradient(position)) {
                *g += gi;
            }
        }

        gradient
    }
}


/// Defines an optimizer that is able to minimize a given objective function.
pub trait Optimizer<F: Function> {
    /// Performs the actual minimization and might return a solution that
    /// is better than the initially provided one.
    fn optimize(&self, objective: &F, x0: Vec<f64>) -> Solution;
}

/*
#[derive(Debug, Clone)]
pub struct Solution {
    /// Position `x` with the lowest corresponding value `f(x)`.
    pub position: Vec<f64>,
    /// The actual value `f(x)`.
    pub value: f64
}
*/


/// A solution of a optimization run.
#[derive(Debug, Clone)]
pub struct Solution {
    /// Found position `x` with the lowest corresponding value `f(x)`.
    pub x: Vec<f64>,
    /// The actual value `f(x)`.
    pub y: f64
}
