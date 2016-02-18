/// Defines an objective function subject for minimization.
pub trait Objective {
    /// Computes the objective function at a given position `x`, i.e., `f(x) = y`.
    fn f(&self, x: &[f64]) -> f64;
}


/// Defines a differentiable objective function.
///
/// In extension to `Objective`, this objective function is able to compute the gradient
/// at a given position.
pub trait DifferentiableObjective: Objective {
    /// Computes the gradient of the objective function at a given position `x`,
    /// i.e., `∀i ∂/∂xᵢ f(x)`.
    fn df(&self, x: &[f64]) -> Vec<f64>;
}


/// Defines an optimizer which is able to minimize a given objective.
pub trait Optimizer<T: Objective> {
    /// Performs the actual minimization and might return a solution that
    /// is better than the initially provided one.
    fn optimize(&self, objective: &T, initial_xs: Vec<f64>) -> Solution;
}


/// A solution of a optimization run.
#[derive(Debug, Clone)]
pub struct Solution {
    /// Found position with the lowest corresponding value `f(x)`.
    pub x: Vec<f64>,
    /// The actual value `f(x)`.
    pub y: f64
}
