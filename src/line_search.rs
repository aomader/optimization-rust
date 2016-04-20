use std::fmt::Debug;

use types::{Function, DifferentiableFunction};


/// Define a line search method, i.e., choosing an appropriate step width.
pub trait LineSearch: Debug {
    /// Performs the actual line search given the current `position` `x` and a `direction` to go to.
    /// Returns the new position.
    fn search<F>(&self, function: &F, initial_position: &[f64], direction: &[f64]) -> Vec<f64>
        where F: DifferentiableFunction;
}


/// Uses a fixed step width `γ` in each iteration instead of performing an actual line search.
#[derive(Debug, Copy, Clone)]
pub struct NoLineSearch {
    fixed_step_width: f64
}

impl NoLineSearch {
    /// Creates a new `FixedStepWidth` given the static step width.
    pub fn new(fixed_step_width: f64) -> NoLineSearch {
        assert!(fixed_step_width > 0.0 && fixed_step_width.is_finite(),
            "fixed_step_width must be greater than 0 and finite");

        NoLineSearch {
            fixed_step_width: fixed_step_width
        }
    }
}

impl LineSearch for NoLineSearch {
    fn search<F>(&self, _function: &F, initial_position: &[f64], direction: &[f64]) -> Vec<f64>
        where F: Function
    {
        initial_position.iter().cloned().zip(direction).map(|(x, d)| {
            x + self.fixed_step_width * d
        }).collect()
    }
}


/// Brute-force line search minimizing the objective function over a set of
/// step width candidates, also known as exact line search.
#[derive(Debug, Copy, Clone)]
pub struct ExactLineSearch {
    start_step_width: f64,
    stop_step_width: f64,
    increase_factor: f64
}

impl ExactLineSearch {
    /// Creates a new `ExactLineSearch` given the `start_step_width`, the `stop_step_width`
    /// and the `increase_factor`. The set of evaluated step widths `γ` is specified as
    /// `{ γ | γ = start_step_width · increase_factorⁱ, i ∈ N, γ <= stop_step_width }`,
    /// assuming that `start_step_width` < `stop_step_width` and `increase_factor` > 1.
    pub fn new(start_step_width: f64, stop_step_width: f64, increase_factor: f64) ->
        ExactLineSearch
    {
        assert!(start_step_width > 0.0 && start_step_width.is_finite(),
            "start_step_width must be greater than 0 and finite");
        assert!(stop_step_width > start_step_width && stop_step_width.is_finite(),
            "stop_step_width must be greater than start_step_width");
        assert!(increase_factor > 1.0 && increase_factor.is_finite(),
            "increase_factor must be greater than 1 and finite");

        ExactLineSearch {
            start_step_width: start_step_width,
            stop_step_width: stop_step_width,
            increase_factor: increase_factor
        }
    }
}

impl LineSearch for ExactLineSearch {
    fn search<F>(&self, function: &F, initial_position: &[f64], direction: &[f64]) -> Vec<f64>
        where F: DifferentiableFunction
    {
        let mut min_position = initial_position.iter().cloned().collect();
        let mut min_value = function.value(initial_position);

        let mut step_width = self.start_step_width;

        loop {
            let position: Vec<_> = initial_position.iter().cloned().zip(direction).map(|(x, d)| {
                x + step_width * d
            }).collect();
            let value = function.value(&position);

            if value < min_value {
                min_position = position;
                min_value = value;
            }

            step_width *= self.increase_factor;

            if step_width >= self.stop_step_width {
                break;
            }
        }

        min_position
    }
}


/// Backtracking line search evaluating the Armijo rule at each step width.
#[derive(Debug, Copy, Clone)]
pub struct ArmijoLineSearch {
    control_parameter: f64,
    initial_step_width: f64,
    decay_factor: f64
}

impl ArmijoLineSearch {
    /// Creates a new `ArmijoLineSearch` given the `control_parameter` ∈ (0, 1), the
    /// `initial_step_width` > 0 and the `decay_factor` ∈ (0, 1).
    ///
    /// Armijo used in his paper the values 0.5, 1.0 and 0.5, respectively.
    pub fn new(control_parameter: f64, initial_step_width: f64, decay_factor: f64) ->
        ArmijoLineSearch
    {
        assert!(control_parameter > 0.0 && control_parameter < 1.0,
            "control_parameter must be in range (0, 1)");
        assert!(initial_step_width > 0.0 && initial_step_width.is_finite(),
            "initial_step_width must be > 0 and finite");
        assert!(decay_factor > 0.0 && decay_factor < 1.0, "decay_factor must be in range (0, 1)");

        ArmijoLineSearch {
            control_parameter: control_parameter,
            initial_step_width: initial_step_width,
            decay_factor: decay_factor
        }
    }
}

impl LineSearch for ArmijoLineSearch {
    fn search<F>(&self, function: &F, initial_position: &[f64], direction: &[f64]) -> Vec<f64>
        where F: DifferentiableFunction
    {
        let initial_value = function.value(initial_position);
        let gradient = function.gradient(initial_position);

        let m = gradient.iter().zip(direction).map(|(g, d)| g * d).sum::<f64>();
        let t = -self.control_parameter * m;

        assert!(t > 0.0);

        let mut step_width = self.initial_step_width;

        loop {
            let position: Vec<_> = initial_position.iter().cloned().zip(direction).map(|(x, d)| {
                x + step_width * d
            }).collect();
            let value = function.value(&position);

            if value <= initial_value - step_width * t {
                return position;
            }

            step_width *= self.decay_factor;
        }
    }
}
