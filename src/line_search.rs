use types::{Objective, DifferentiableObjective};


/// Define a line search method, i.e., choosing an appropriate step width.
pub trait LineSearch {
    /// Performs the actual line search given the current position `xs` and a `direction` to go to.
    /// Should return the new position and the corresponding value.
    fn search<T: DifferentiableObjective>(&self, objective: &T, x: &[f64], direction: &[f64]) -> (Vec<f64>, f64);
}


/// Uses a fixed step width `γ` in each iteration instead of performing an actual line search.
#[derive(Debug, Copy, Clone)]
pub struct NoLineSearch {
    step_width: f64
}

impl NoLineSearch {
    /// Creates a new `FixedStepWidth` given the static step width.
    pub fn new(step_width: f64) -> NoLineSearch {
        NoLineSearch {
            step_width: step_width
        }
    }
}

impl LineSearch for NoLineSearch {
    fn search<T: Objective>(&self, objective: &T, x: &[f64], direction: &[f64]) -> (Vec<f64>, f64) {
        //debug!("Use fixed step width {:e}", self.step_width);

        let x: Vec<_> = x.iter().cloned().zip(direction).map(|(x, d)|
            x + self.step_width * d ).collect();
        let y = objective.value(&x);
        (x, y)
    }
}


/// Brute-force line search minimizing the objective function over a set of
/// step width candidates.
#[derive(Debug, Copy, Clone)]
pub struct ExactLineSearch {
    start: f64,
    stop: f64,
    factor: f64
}

impl ExactLineSearch {
    /// Creates a new `ExactLineSearch` given the `start` value, the `stop` value and the
    /// `increment`. The set of evaluated step widths is defined as
    /// `{ γ | γ = start * factorⁱ, i ∈ N, γ <= stop }`
    /// assuming that `start` < `stop` and `factor` > 1.
    pub fn new(start: f64, stop: f64, factor: f64) -> ExactLineSearch {
        assert!(start > 0.0);
        assert!(stop > start);
        assert!(factor > 1.0);

        ExactLineSearch {
            start: start,
            stop: stop,
            factor: factor
        }
    }
}

impl LineSearch for ExactLineSearch {
    fn search<T: DifferentiableObjective>(&self, objective: &T, x0: &[f64], direction: &[f64]) -> (Vec<f64>, f64) {
        let mut min_x: Vec<_> = x0.iter().cloned().collect();
        let mut min_y = objective.value(&min_x);

        let mut γ = self.start;

        loop {
            let x: Vec<_> = x0.iter().cloned().zip(direction).map(|(x, &d)| {
                x + γ * d
            }).collect();
            let y = objective.value(&x);

            if y < min_y {
                min_x = x;
                min_y = y;
            }

            γ *= self.factor;

            if γ > self.stop {
                break;
            }
        }

        (min_x, min_y)
    }
}


/// Backtracking line search evaluating the Armijo rule at each step width.
#[derive(Debug, Copy, Clone)]
pub struct ArmijoLineSearch {
    c: f64,
    γ: f64,
    decay: f64
}

impl ArmijoLineSearch {
    /// Creates a new `ArmijoLineSearch` given the control parameter `c` ∈ (0, 1), the
    /// initial step width `γ` > 0 and the decay factor `decay` ∈ (0, 1).
    ///
    /// In the paper by Armijo he used the values 0.5, 1.0 and 0.5, respectively.
    pub fn new(c: f64, γ: f64, decay: f64) -> ArmijoLineSearch {
        assert!(c > 0.0 && c < 1.0);
        assert!(γ > 0.0);
        assert!(decay > 0.0 && decay < 1.0);

        ArmijoLineSearch {
            c: c,
            γ: γ,
            decay: decay
        }
    }
}

impl LineSearch for ArmijoLineSearch {
    fn search<T: DifferentiableObjective>(&self, objective: &T, xs: &[f64], direction: &[f64]) -> (Vec<f64>, f64) {
        let ynull = objective.value(xs);
        let gradient = objective.gradient(xs);

        let m = gradient.iter().zip(direction).map(|(g, d)| g * d).sum::<f64>();
        let t = -self.c * m;

        assert!(t > 0.0);

        let mut γ = self.γ;

        loop {
            let xs: Vec<_> = xs.iter().cloned().zip(direction).map(|(x, &d)| {
                x + γ * d
            }).collect();
            let y = objective.value(&xs);

            if y <= ynull - γ * t {
                return (xs, y);
            }

            γ *= self.decay;
        }
    }
}
