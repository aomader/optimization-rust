//! Collection of various optimization algorithms and strategies.
//!
//!
//!

#![feature(iter_arith)]

// lints
//#![feature(plugin)]
//#![plugin(clippy)]
//#![plugin(herbie_lint)]

#[macro_use]
extern crate log;

extern crate rand;


#[cfg(any(test, feature="problems"))]
#[macro_use]
pub mod problems;

mod types;
mod utils;
mod line_search;

pub mod gd;
pub mod sgd;


// public re-exports
pub use types::{Function, DifferentiableFunction, Summation, Minimizer, Evaluation};
pub use line_search::{LineSearch, NoLineSearch, ExactLineSearch, ArmijoLineSearch};
pub use gd::GradientDescent;
pub use sgd::StochasticGradientDescent;
