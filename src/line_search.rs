use types::{Function, DifferentiableFunction};


/// Define a line search method, i.e., choosing an appropriate step width.
pub trait LineSearch {
    /// Performs the actual line search given the current `position` `x` and a `direction` to go to.
    /// Returns the new position and the corresponding value, in that order.
    fn search<F: DifferentiableFunction>(&self, objective: &F, position: &[f64], direction: &[f64])
        -> (Vec<f64>, f64);
}


/// Uses a fixed step width `γ` in each iteration instead of performing an actual line search.
#[derive(Debug, Copy, Clone)]
pub struct NoLineSearch {
    step_width: f64
}

impl NoLineSearch {
    /// Creates a new `FixedStepWidth` given the static step width.
    pub fn new(step_width: f64) -> NoLineSearch {
        assert!(step_width.is_finite() && step_width > 0.0, "step_width must be finite and > 0");

        NoLineSearch {
            step_width: step_width
        }
    }
}

impl LineSearch for NoLineSearch {
    fn search<F: Function>(&self, objective: &F, x: &[f64], direction: &[f64]) ->
        (Vec<f64>, f64)
    {
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
    /// `{ γ | γ = start · factorⁱ, i ∈ N, γ <= stop }`
    /// assuming that `start` < `stop` and `factor` > 1.
    pub fn new(start: f64, stop: f64, factor: f64) -> ExactLineSearch {
        assert!(start > 0.0, "start must be > 0");
        assert!(stop > start, "");
        assert!(factor > 1.0);

        ExactLineSearch {
            start: start,
            stop: stop,
            factor: factor
        }
    }
}

impl LineSearch for ExactLineSearch {
    fn search<F: DifferentiableFunction>(&self, objective: &F, x0: &[f64], direction: &[f64]) -> (Vec<f64>, f64) {
        let mut min_x: Vec<_> = x0.iter().cloned().collect();
        let mut min_y = objective.value(&min_x);

        let mut step_width = self.start;

        loop {
            let x: Vec<_> = x0.iter().cloned().zip(direction).map(|(x, &d)| {
                x + step_width * d
            }).collect();
            let y = objective.value(&x);

            if y < min_y {
                min_x = x;
                min_y = y;
            }

            step_width *= self.factor;

            if step_width > self.stop {
                break;
            }
        }

        (min_x, min_y)
    }
}


/// Backtracking line search evaluating the Armijo rule at each step width.
#[derive(Debug, Copy, Clone)]
pub struct ArmijoLineSearch {
    control: f64,
    start: f64,
    decay: f64
}

impl ArmijoLineSearch {
    /// Creates a new `ArmijoLineSearch` given the `control` parameter ∈ (0, 1), the
    /// initial step width `start` > 0 and the `decay` factor ∈ (0, 1).
    ///
    /// In the paper by Armijo he used the values 0.5, 1.0 and 0.5, respectively.
    pub fn new(control: f64, start: f64, decay: f64) -> ArmijoLineSearch {
        assert!(control > 0.0 && control < 1.0, "control must be in range (0, 1)");
        assert!(start > 0.0, "start must be > 0");
        assert!(decay > 0.0 && decay < 1.0, "decay must be in range (0, 1)");

        ArmijoLineSearch {
            control: control,
            start: start,
            decay: decay
        }
    }
}

impl LineSearch for ArmijoLineSearch {
    fn search<F: DifferentiableFunction>(&self, objective: &F, xs: &[f64], direction: &[f64]) -> (Vec<f64>, f64) {
        let ynull = objective.value(xs);
        let gradient = objective.gradient(xs);

        let m = gradient.iter().zip(direction).map(|(g, d)| g * d).sum::<f64>();
        let t = -self.control * m;

        assert!(t > 0.0);

        let mut step_width = self.start;

        loop {
            let xs: Vec<_> = xs.iter().cloned().zip(direction).map(|(x, &d)| {
                x + step_width * d
            }).collect();
            let y = objective.value(&xs);

            if y <= ynull - step_width * t {
                return (xs, y);
            }

            step_width *= self.decay;
        }
    }
}
