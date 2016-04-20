use std::f64::EPSILON;

use types::{Function, DifferentiableFunction};


const EPSILON: f64 = 2.2204460492503131e-6;




/// Wraps a function for which to provide numeric gradient computation.
pub struct NumericDifferentiation<F: Function> {
    function: F
}

impl<F: Function> NumericalDifferentiation<F> {
    pub fn new(function: F) -> Self {
        NumericalDifferentiation {
            function: function
        }
    }
}

impl<F: Function> Function for NumericDifferentiation<F> {
    fn value(&self, position: &[f64]) -> f64 {
        self.function.value(position)
    }
}

impl<F: Function> DifferentiableObjective for NumericDifferentiation<F> {
    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        let mut x2: Vec<_> = x.iter().cloned().collect();

        const h: f64 = EPSILON;
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
