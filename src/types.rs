use std::borrow::Borrow;
use std::ops::Deref;


/// Defines an objective function `f` that is subject to minimization.
///
/// For convenience every function with the same signature as `value()` qualifies as
/// an objective function, e.g., minimizing a closure is perfectly fine.
pub trait Function {
    /// Computes the objective function at a given `position` `x`, i.e., `f(x) = y`.
    fn value(&self, position: &[f64]) -> f64;
}

impl<F: Fn(&[f64]) -> f64> Function for F {
    fn value(&self, position: &[f64]) -> f64 {
        self(position)
    }
}


/// Defines an objective function `f` that is able to compute the first derivative
/// `f'(x)` analytically.
pub trait Derivative1: Function {
    /// Computes the gradient of the objective function at a given `position` `x`,
    /// i.e., `∀ᵢ ∂/∂xᵢ f(x) = ∇f(x)`.
    fn gradient(&self, position: &[f64]) -> Vec<f64>;
}


/// Represents a summation of different functions.
pub trait SummationFunction: Function {
    /// Number of terms of the summation.
    fn terms(&self) -> usize;

    /// Computes the partial value over a set of `terms` at the given `position`.
    fn partial_value(&self, position: &[f64], terms: &[usize]) -> f64;
}


/// Represents a summation of function that support the computation of the first derivative.
pub trait SummationDerivative1: SummationFunction + Derivative1 {
    /// Computes the partial gradient over a set of `terms` at the given `position`.
    fn partial_gradient(&self, position: &[f64], terms: &[usize]) -> Vec<f64>;
}


/// New-type to support summation over common collection types.
pub struct Summation<T>(T);

impl<C: Deref<Target=[F]>, F: Function> Function for Summation<C> {
    fn value(&self, position: &[f64]) -> f64 {
        let mut value = 0.0;

        for term in &*self.0 {
            value += term.value(position);
        }

        value
    }
}

impl<C: Deref<Target=[F]>, F: Function> SummationFunction for Summation<C> {
    fn terms(&self) -> usize {
        self.0.len()
    }

    fn partial_value(&self, position: &[f64], terms: &[usize]) -> f64 {
        let mut value = 0.0;

        for &term in terms {
            value += self.0[term].value(position);
        }

        value
    }
}

impl<C: Deref<Target=[F]>, F: Derivative1> Derivative1 for Summation<C> {
    fn gradient(&self, position: &[f64]) -> Vec<f64> {
        let mut gradient = vec![0.0; position.len()];

        for term in &*self.0 {
            for (g, gi) in gradient.iter_mut().zip(term.gradient(position)) {
                *g += gi;
            }
        }

        gradient
    }
}

impl<C: Deref<Target=[F]>, F: Derivative1> SummationDerivative1 for Summation<C> {
    fn partial_gradient(&self, position: &[f64], terms: &[usize]) -> Vec<f64>
    {
        let mut gradient = vec![0.0; position.len()];

        // TODO: This can be optimized easily
        for &term in terms {
            for (g, gi) in gradient.iter_mut().zip(self.0[term].gradient(position)) {
                *g += gi;
            }
        }

        gradient
    }
}


/// Defines an optimizer that is able to minimize a given objective function `F`.
pub trait Minimizer<F: ?Sized> {
    type Solution: Evaluation;

    /// Performs the actual minimization and returns a solution that
    /// might be better than the initially provided one.
    fn minimize(&self, function: &F, initial_position: Vec<f64>) -> Self::Solution;
}


/// Captures the essence of a function evaluation.
pub trait Evaluation {
    /// Position `x` with the lowest corresponding value `f(x)`.
    fn position(&self) -> &[f64];

    /// The actual value `f(x)`.
    fn value(&self) -> f64;
}


/// A solution of a minimization run providing only the minimal information.
///
/// Each `Minimizer` might yield different types of solution structs which provide more
/// information.
#[derive(Debug, Clone)]
pub struct Solution {
    /// Position `x` of the lowest corresponding value `f(x)` that has been found.
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


#[cfg(test)]
mod tests {
    use super::{Function, Summation, SummationFunction};

    pub struct Constant(f64);

    impl Function for Constant {
        fn value(&self, _position: &[f64]) -> f64 {
            self.0
        }
    }

    #[test]
    fn test_summation_function_value() {
        let summation = Summation(vec![Constant(1.0), Constant(2.0), Constant(-4.0)]);
        assert_eq!(summation.value(&[]), -1.0);
    }

    #[test]
    fn test_summation_function_partial_value() {
        let summation = Summation(vec![Constant(1.0), Constant(2.0), Constant(-4.0)]);
        assert_eq!(summation.partial_value(&[], &[0, 1]), 3.0);
        assert_eq!(summation.partial_value(&[], &[0, 2]), -3.0);
    }
}
