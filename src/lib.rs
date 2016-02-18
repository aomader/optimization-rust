//! Collection of various optimization algorithms and strategies.
//!
//!
//!

#![feature(plugin)]
#![feature(iter_arith)]
#![feature(non_ascii_idents)]

// lints
//#![plugin(clippy)]
//#![plugin(herbie_lint)]

#[macro_use]
extern crate log;

// private modules
mod types;
mod line_search;
mod gradient_descent;
mod utils;

// public re-exports
pub use types::{Objective, DifferentiableObjective, Optimizer};
pub use line_search::{LineSearch, NoLineSearch, ExactLineSearch, ArmijoLineSearch};
pub use gradient_descent::{GradientDescent};
