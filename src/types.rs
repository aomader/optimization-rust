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


/// Defines an optimizer that is able to minimize a given objective function `F`.
pub trait Minimizer<F: Function> {
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
