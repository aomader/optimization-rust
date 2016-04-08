use std::f64::EPSILON;

use types::{Objective, DifferentiableObjective};

pub struct Numeric<T: Objective> {
    objective: T
}

impl<T: Objective> Numeric<T> {
    pub fn new(objective: T) -> Self {
        Numeric {
            objective: objective
        }
    }
}

impl<T: Objective> Objective for Numeric<T> {
    fn value(&self, x: &[f64]) -> f64 {
        self.objective.value(x)
    }
}

impl<T: Objective> DifferentiableObjective for Numeric<T> {
    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        let mut x2: Vec<_> = x.iter().cloned().collect();

        const h: f64 = 1.0;// 2.2204460492503131e-10;
        const h2: f64 = h + h;


        (0..x.len()).map(|i| {
            x2[i] = x[i] + h;
            let upper = self.objective.value(&x2);

            x2[i] = x[i] - h;
            let lower = self.objective.value(&x2);

            x2[i] = x[i];

            (upper - lower) / h2
        }).collect()
    }
}
