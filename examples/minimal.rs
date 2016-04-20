//! Minimal usage example of the library.
//!
//! Run with `cargo run --example minimal --features problems`.


extern crate optimization;


use optimization::{Minimizer, Evaluation, GradientDescent};
use optimization::problems::{Problem, Rosenbrock};


pub fn main() {
    // the target function we want to minimize, for educational reasons we use
    // the Rosenbrock function
    let function = Rosenbrock::default();

    // we use a simple gradient descent scheme
    let minimizer = GradientDescent::new();

    // perform the actual minimization, depending on the task this may take some time
    // it may be useful to install a log sink to seewhat's going on
    let solution = minimizer.minimize(&function, function.random_start());

    println!("Found solution for Rosenbrock function at f({:?}) = {:?}",
        solution.position(), solution.value());
}
