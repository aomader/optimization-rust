/// Defines an objective function subject for minimization.
pub trait Objective {
    /// Computes the objective function at a given position `x`, i.e., `f(x) = y`.
    fn value(&self, x: &[f64]) -> f64;
}


/// Defines a differentiable objective function.
///
/// In extension to `Objective`, this objective function is able to compute the gradient
/// at a given position.
pub trait DifferentiableObjective: Objective {
    /// Computes the gradient of the objective function at a given position `x`,
    /// i.e., `∀i ∂/∂xᵢ f(x)`.
    fn gradient(&self, x: &[f64]) -> Vec<f64>;
}


pub trait SampledObjective {

    fn samples(&self) -> usize;

    fn sample_value(&self, x: &[f64], i: usize) -> f64;

}

impl<T: ?Sized + SampledObjective> Objective for T {
    fn value(&self, x: &[f64]) -> f64 {
        let mut value = 0.0;
        for i in 0..self.samples() {
            value += self.sample_value(x, i);
        }
        value
    }
}

pub trait SampledDifferentiableObjective: SampledObjective {
    fn sample_gradient(&self, x: &[f64], i: usize) -> Vec<f64>;
}

impl<T: ?Sized + SampledDifferentiableObjective> DifferentiableObjective for T {
    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        let mut gradient = vec![0.0; x.len()];
        for i in 0..self.samples() {
            for (g, gi) in gradient.iter_mut().zip(self.sample_gradient(x, i)) {
                *g += gi;
            }
        }
        gradient
    }
}


/// Defines an optimizer which is able to minimize a given objective.
pub trait Optimizer<T: Objective> {
    /// Performs the actual minimization and might return a solution that
    /// is better than the initially provided one.
    fn optimize(&self, objective: &T, x0: Vec<f64>) -> Solution;
}


/// A solution of a optimization run.
#[derive(Debug, Clone)]
pub struct Solution {
    /// Found position `x` with the lowest corresponding value `f(x)`.
    pub x: Vec<f64>,
    /// The actual value `f(x)`.
    pub y: f64
}
