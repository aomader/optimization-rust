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


/// Defines an optimizer that is able to minimize a given objective function `F`.
pub trait Minimizer<F: Function> {
    type Solution: Evaluation;

    /// Performs the actual minimization and returns a solution that
    /// might be better than the initially provided one.
    fn minimize(&self, function: &F, initial_position: Vec<f64>) -> Self::Solution;
}

/*
pub trait IterativeMinimizer: Minimizer<F> {
    fn max_iterations(&mut self, max_iterations: Option<u64>) -> &mut Self;

    //fn iteration_callback(&mut self, callback: &FnMut) -> &mut Self;
}
*/


/// Captures the essence of a function evaluation.
pub trait Evaluation {
    /// Position `x` with the lowest corresponding value `f(x)`.
    fn position(&self) -> &[f64];

    /// The actual value `f(x)`.
    fn value(&self) -> f64;
}


/// A solution of a minimization run providing only the minimal information.
#[derive(Debug, Clone)]
pub struct Solution {
    /// Position `x` with the lowest corresponding value `f(x)`.
    pub position: Vec<f64>,
    /// The actual value `f(x)`.
    pub value: f64
}

impl Solution {
    /// Creates a new `Solution` given the `position` as well as the corresponding `value`.
    pub fn new(position: Vec<f64>, value: f64) -> Solution {
        Solution {
            position: position,
            value: value
        }
    }
}

impl Evaluation for Solution {
    fn position(&self) -> &[f64] {
        &self.position
    }

    fn value(&self) -> f64 {
        self.value
    }
}
