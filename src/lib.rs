//! Collection of various optimization algorithms and strategies.
//!
//! # Building Blocks
//!
//! Each central primitive is specified by a trait:
//!
//! - **`Function`** - Specifies a function that can be minimized
//! - **`Derivative1`** - Extends a `Function` by the analytical first derivative
//! - **`NumericalDifferentiation`** - Provides numerical differentiation for arbitrary `Function`s
//! - **`Minimizer`** - A minimization algorithm
//! - **`Evaluation`** - A function evaluation `f(x) = y` that is returned by a `Minimizer`
//!
//! # Algorithms
//!
//! Currently, the following algorithms are implemented. This list is not final and being
//! expanded over time.
//!
//! - **`GradientDescent`** - Iterative gradient descent minimization, supporting various line
//!   search methods:
//!    - *`FixedStepWidth`* - No line search is performed, but a fixed step width is used
//!    - *`ExactLineSearch`* - Exhaustive line search over a set of step widths
//!    - *`ArmijoLineSearch`* - Backtracking line search using the Armijo rule as stopping
//!      criterion


#![cfg_attr(feature = "unstable", feature(plugin))]
#![cfg_attr(feature = "unstable", plugin(clippy))]
#![cfg_attr(feature = "unstable", plugin(herbie_lint))]


#[macro_use]
extern crate log;
extern crate rand;


#[macro_use]
pub mod problems;

mod types;
mod utils;
mod numeric;
mod line_search;
mod gd;


pub use types::{Function, Derivative1, Minimizer, Evaluation};
pub use numeric::NumericalDifferentiation;
pub use line_search::{LineSearch, FixedStepWidth, ExactLineSearch, ArmijoLineSearch};
pub use gd::GradientDescent;
