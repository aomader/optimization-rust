//! Illustration of fitting a linear regression model using stochastic gradient descent
//! given a few noisy sample observations.
//!
//! Run with `cargo run --example line_fitting`.


#![allow(clippy::many_single_char_names)]


extern crate env_logger;
extern crate rand;
extern crate rand_distr;

extern crate optimization;


use std::f64::consts::PI;
use rand::prelude::*;
use rand_distr::StandardNormal;

use optimization::*;


fn main() {
    env_logger::init();

    // the true coefficients of our linear model
    let true_coefficients = &[13.37, -4.2, PI];

    println!("Trying to approximate the true linear regression coefficients {:?} using SGD \
        given 100 noisy samples", true_coefficients);

    let noisy_observations = (0..100).map(|_| {
        let x = random::<[f64; 2]>();
        let noise: f64 = thread_rng().sample(StandardNormal);
        let y = linear_regression(true_coefficients, &x) + noise;

        (x.to_vec(), y)
    }).collect();


    // the actual function we want to minimize, which in our case corresponds to the
    // sum squared error
    let sse = SSE {
        observations: noisy_observations
    };

    let solution = StochasticGradientDescent::new()
        .max_iterations(Some(1000))
        .minimize(&sse, vec![1.0; true_coefficients.len()]);

    println!("Found coefficients {:?} with a SSE = {:?}", solution.position, solution.value);
}


// the sum squared error measure we want to minimize over a set of observations
struct SSE {
    observations: Vec<(Vec<f64>, f64)>
}

impl Summation for SSE {
    fn terms(&self) -> usize {
        self.observations.len()
    }

    fn term_value(&self, w: &[f64], i: usize) -> f64 {
        let (ref x, y) = self.observations[i];

        0.5 * (y - linear_regression(w, x)).powi(2)
    }
}

impl Summation1 for SSE {
    fn term_gradient(&self, w: &[f64], i: usize) -> Vec<f64> {
        let (ref x, y) = self.observations[i];

        let e = y - linear_regression(w, x);

        let mut gradient = vec![e * -1.0];

        for x in x {
            gradient.push(e * -x);
        }

        gradient
    }
}


// a simple linear regression model, i.e., f(x) = w_0 + w_1*x_1 + w_2*x_2 + ...
fn linear_regression(w: &[f64], x: &[f64]) -> f64 {
    let mut y = w[0];

    for (w, x) in w[1..].iter().zip(x) {
        y += w * x;
    }

    y
}
