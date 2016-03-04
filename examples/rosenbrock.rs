#![feature(iter_arith)]

extern crate log;
extern crate env_logger;
extern crate optimization;

use log::{LogLevelFilter};
use env_logger::LogBuilder;

use optimization::*;


pub fn main() {
    LogBuilder::new()
        .filter(None, LogLevelFilter::Debug)
        .init();

    let problem = Rosenbrock {
        a: 1.0,
        b: 100.0
    };

    let solution = GradientDescent::new()
        .optimize(&problem, vec![-3.0, -4.0]);

    println!("Solution for Rosenbrock function: {:?}", solution);
}


struct Rosenbrock {
    a: f64,
    b: f64
}

impl Objective for Rosenbrock {
    fn value(&self, xs: &[f64]) -> f64 {
        assert!(xs.len() == 2);

        (self.a - xs[0]).powi(2) + self.b * (xs[1] - xs[0].powi(2)).powi(2)
    }
}

impl DifferentiableObjective for Rosenbrock {
    fn gradient(&self, xs: &[f64]) -> Vec<f64> {
        assert!(xs.len() == 2);

        let dx = -2.0 * self.a + 4.0 * self.b * xs[0].powi(3) - 4.0 * self.b * xs[0] * xs[1] + 2.0 * xs[0];
        let dy = 2.0 * self.b * (xs[1] - xs[0].powi(2));

        vec![dx, dy]
    }
}
