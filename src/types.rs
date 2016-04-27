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


pub struct Func<F: Fn(&[f64]) -> f64>(pub F);

impl<F: Fn(&[f64]) -> f64> Function for Func<F> {
    fn value(&self, position: &[f64]) -> f64 {
        self.0(position)
    }
}


/// Defines an objective function `f` that is able to compute the first derivative
/// `f'(x)` analytically.
pub trait Derivative1: Function {
    /// Computes the gradient of the objective function at a given `position` `x`,
    /// i.e., `∀ᵢ ∂/∂xᵢ f(x) = ∇f(x)`.
    fn gradient(&self, position: &[f64]) -> Vec<f64>;
}


/// Defines a summation of individual functions, i.e., f(x) = ∑ᵢ fᵢ(x).
pub trait Summation: Function {
    /// Returns the number of individual functions that are terms of the summation.
    fn terms(&self) -> usize;

    /// Comptues the value of one individual function indentified by its index `term`,
    /// given the `position` `x`.
    fn term_value(&self, position: &[f64], term: usize) -> f64;

    /// Computes the partial sum over a set of individual functions identified by `terms`.
    fn partial_value<T: IntoIterator<Item=I>, I: Borrow<usize>>(&self, position: &[f64], terms: T) -> f64 {
        let mut value = 0.0;

        for term in terms {
            value += self.term_value(position, *term.borrow());
        }

        value
    }
}

impl<S: Summation> Function for S {
    fn value(&self, position: &[f64]) -> f64 {
        self.partial_value(position, 0..self.terms())
    }
}


/// Defines a summation of individual functions `fᵢ(x)`, assuming that each function has a first
/// derivative.
pub trait Summation1: Summation + Derivative1 {
    /// Computes the gradient of one individual function identified by `term` at the given
    /// `position`.
    fn term_gradient(&self, position: &[f64], term: usize) -> Vec<f64>;

    /// Computes the partial gradient over a set of `terms` at the given `position`.
    fn partial_gradient<T: IntoIterator<Item=I>, I: Borrow<usize>>(&self, position: &[f64], terms: T) -> Vec<f64> {
        let mut gradient = vec![0.0; position.len()];

        for term in terms {
            for (g, gi) in gradient.iter_mut().zip(self.term_gradient(position, *term.borrow())) {
                *g += gi;
            }
        }

        gradient
    }
}

impl<S: Summation1> Derivative1 for S {
    fn gradient(&self, position: &[f64]) -> Vec<f64> {
        self.partial_gradient(position, 0..self.terms())
    }
}


/// New-type to support summation over common collection types without requiring to
/// implement `Summation` for custom types.
pub struct Sum<T>(pub T);

impl<C: Deref<Target=[F]>, F: Function> Summation for Sum<C> {
    fn terms(&self) -> usize {
        self.0.len()
    }

    fn term_value(&self, position: &[f64], term: usize) -> f64 {
        self.0[term].value(position)
    }
}

impl<C: Deref<Target=[F]>, F: Derivative1> Summation1 for Sum<C> {
    fn term_gradient(&self, position: &[f64], term: usize) -> Vec<f64> {
        self.0[term].gradient(position)
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
    use super::{Function, Sum, Summation};

    pub struct Constant(f64);

    impl Function for Constant {
        fn value(&self, _position: &[f64]) -> f64 {
            self.0
        }
    }

    #[test]
    fn test_sum_value() {
        let summation = Sum(vec![Constant(1.0), Constant(2.0), Constant(-4.0)]);
        assert_eq!(summation.value(&[]), -1.0);
    }

/*
    #[test]
    fn test_sum_partial_value() {
        let summation = Sum(&[Constant(1.0), Constant(2.0), Constant(-4.0)]);
        assert_eq!(summation.partial_value(&[], &[0, 1]), 3.0);
        assert_eq!(summation.partial_value(&[], &[0, 2]), -3.0);
    }*/
}
