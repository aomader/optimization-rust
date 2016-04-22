//! Minimizing the Rosenbrock function using Gradient Descent by applying
//! numerical differentiation.
//!
//! Run with `cargo run --example minimal`.


extern crate env_logger;

extern crate optimization;


use optimization::{Minimizer, GradientDescent, NumericalDifferentiation};


pub fn main() {
    let _ = env_logger::init();

    // the target function we want to minimize, for educational reasons we use
    // the Rosenbrock function
    let function = NumericalDifferentiation::new(|x: &[f64]| {
        (1.0 - x[0]).powi(2) + 100.0*(x[1] - x[0].powi(2)).powi(2)
    });

    // we use a simple gradient descent scheme
    let minimizer = GradientDescent::new();

    // perform the actual minimization, depending on the task this may take some time
    // it may be useful to install a log sink to seew hat's going on
    let solution = minimizer.minimize(&function, vec![-3.0, -4.0]);

    println!("Found solution for Rosenbrock function at f({:?}) = {:?}",
        solution.position, solution.value);
}
