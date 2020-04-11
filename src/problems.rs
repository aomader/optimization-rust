//! Common optimization problems for testing purposes.
//!
//! Currently, the following [optimization test functions] are implemented.
//!
//! ## Bowl-Shaped
//!
//! * [`Sphere`](http://www.sfu.ca/~ssurjano/spheref.html)
//!
//! ## Valley-Shaped
//!
//! * [`Rosenbrock`](http://www.sfu.ca/~ssurjano/rosen.html)
//!
//! [optimization test functions]: http://www.sfu.ca/~ssurjano/optimization.html

use rand::random;
use std::f64::INFINITY;
use std::ops::Add;

use types::{Function, Function1};


/// Specifies a well known optimization problem.
pub trait Problem: Function + Default {
    /// Returns the dimensionality of the input domain.
    fn dimensions(&self) -> usize;

    /// Returns the input domain of the function in terms of upper and lower,
    /// respectively, for each input dimension.
    fn domain(&self) -> Vec<(f64, f64)>;

    /// Returns the position as well as the value of the global minimum.
    fn minimum(&self) -> (Vec<f64>, f64);

    /// Generates a random and **feasible** position to start a minimization.
    fn random_start(&self) -> Vec<f64>;

    /// Tests whether the supplied position is legal for this function.
    fn is_legal_position(&self, position: &[f64]) -> bool {
        position.len() == self.dimensions() &&
        position.iter().zip(self.domain()).all(|(&x, (lower, upper))| {
            lower < x && x < upper
        })
    }
}


macro_rules! define_problem {
    ( $name:ident: $this:ident,
        default: $def:expr,
        dimensions: $dims:expr,
        domain: $domain:expr,
        minimum: $miny:expr,
        at: $minx:expr,
        start: $start:expr,
        value: $x1:ident => $value:expr,
        gradient: $x2:ident => $gradient:expr ) =>
    {
        impl Default for $name {
            fn default() -> Self {
                $def
            }
        }

        impl Function for $name {
            fn value(&$this, $x1: &[f64]) -> f64 {
                assert!($this.is_legal_position($x1));

                $value
            }
        }

        impl Function1 for $name {
            fn gradient(&$this, $x2: &[f64]) -> Vec<f64> {
                assert!($this.is_legal_position($x2));

                $gradient
            }
        }

        impl Problem for $name {
            fn dimensions(&$this) -> usize {
                $dims
            }

            fn domain(&$this) -> Vec<(f64, f64)> {
                $domain
            }

            fn minimum(&$this) -> (Vec<f64>, f64) {
                ($minx, $miny)
            }

            fn random_start(&$this) -> Vec<f64> {
                $start
            }
        }
    };
}


/// n-dimensional Sphere function.
///
/// It is continuous, convex and unimodal:
///
/// > f(x) = ∑ᵢ xᵢ²
///
/// *Global minimum*: `f(0,...,0) = 0`
#[derive(Debug, Copy, Clone)]
pub struct Sphere {
    dimensions: usize
}

impl Sphere {
    pub fn new(dimensions: usize) -> Sphere {
        assert!(dimensions > 0, "dimensions must be larger than 1");

        Sphere {
            dimensions
        }
    }
}

define_problem!{Sphere: self,
    default: Sphere::new(2),
    dimensions: self.dimensions,
    domain: (0..self.dimensions).map(|_| (-INFINITY, INFINITY)).collect(),
    minimum: 0.0,
    at: (0..self.dimensions).map(|_| 0.0).collect(),
    start: (0..self.dimensions).map(|_| random::<f64>() * 10.24 - 5.12).collect(),
    value: x => x.iter().map(|x| x.powi(2)).fold(0.0, Add::add),
    gradient: x => x.iter().map(|x| 2.0 * x).collect()
}


/// Two-dimensional Rosenbrock function.
///
/// A non-convex function with its global minimum inside a long, narrow, parabolic
/// shaped flat valley:
///
/// > f(x, y) = (a - x)² + b (y - x²)²
///
/// *Global minimum*: `f(a, a²) = 0`
#[derive(Debug, Copy, Clone)]
pub struct Rosenbrock {
    a: f64,
    b: f64
}

impl Rosenbrock {
    /// Creates a new `Rosenbrock` function given `a` and `b`, commonly definied
    /// with 1 and 100, respectively, which also corresponds to the `default`.
    pub fn new(a: f64, b: f64) -> Rosenbrock {
        Rosenbrock {
            a,
            b
        }
    }
}

define_problem!{Rosenbrock: self,
    default: Rosenbrock::new(1.0, 100.0),
    dimensions: 2,
    domain: vec![(-INFINITY, INFINITY), (-INFINITY, INFINITY)],
    minimum: 0.0,
    at: vec![self.a, self.a * self.a],
    start: (0..2).map(|_| random::<f64>() * 4.096 - 2.048).collect(),
    value: x => (self.a - x[0]).powi(2) + self.b * (x[1] - x[0].powi(2)).powi(2),
    gradient: x => vec![-2.0 * self.a + 4.0 * self.b * x[0].powi(3) - 4.0 * self.b * x[0] * x[1] + 2.0 * x[0],
                        2.0 * self.b * (x[1] - x[0].powi(2))]
}


/*
pub struct McCormick;

impl McCormick {
    pub fn new() -> McCormick {
        McCormick
    }
}

define_problem!{McCormick: self,
    default: McCormick::new(),
    dimensions: 2,
    domain: vec![(-INFINITY, INFINITY), (-INFINITY, INFINITY)],
    minimum: -1.9133,
    at: vec![-0.54719, -1.54719],
    start: vec![random::<f64>() * 5.5 - 1.5, random::<f64>() * 7.0 - 3.0],
    value: x => (x[0] + x[1]).sin() + (x[0] - x[1]).powi(2) - 1.5 * x[0] + 2.5 * x[1] + 1.0,
    gradient: x => vec![(x[0] + x[1]).cos() + 2.0 * (x[0] - x[1]) - 1.5,
                        (x[0] + x[1]).cos() - 2.0 * (x[0] - x[1]) + 2.5]
}
*/


#[cfg(test)]
macro_rules! test_minimizer {
    ( $minimizer:expr, $( $name:ident => $problem:expr ),* ) => {
        $(
            #[test]
            fn $name() {
                let minimizer = $minimizer;
                let problem = $problem;

                for _ in 0..100 {
                    let position = $crate::problems::Problem::random_start(&problem);

                    let solution = $crate::Minimizer::minimize(&minimizer,
                        &problem, position);

                    let distance = $crate::Evaluation::position(&solution).iter()
                        .zip($crate::problems::Problem::minimum(&problem).0)
                        .map(|(a, b)| (a - b).powi(2))
                        .fold(0.0, ::std::ops::Add::add)
                        .sqrt();

                    assert!(distance < 1.0e-2);
                }
            }
        )*
    };
}
