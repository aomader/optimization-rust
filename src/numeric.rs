use std::f64::EPSILON;

use problems::Problem;
use types::{Function, Derivative1, Func};


/// Wraps a function for which to provide numeric differentiation.
///
/// Uses simple one step forward finite difference with step width `h = √εx`.
///
/// # Examples
///
/// ```
/// # use self::optimization::*;
/// let square = NumericalDifferentiation::new(Func(|x: &[f64]| {
///     x[0] * x[0]
/// }));
///
/// assert!(square.gradient(&[0.0])[0] < 1.0e-3);
/// assert!(square.gradient(&[1.0])[0] > 1.0);
/// assert!(square.gradient(&[-1.0])[0] < 1.0);
/// ```
pub struct NumericalDifferentiation<F: Function> {
    function: F
}

impl<F: Function> NumericalDifferentiation<F> {
    /// Creates a new differentiable function by using the supplied `function` in
    /// combination with numeric differentiation to find the derivatives.
    pub fn new(function: F) -> Self {
        NumericalDifferentiation {
            function: function
        }
    }
}

impl<F: Function> Function for NumericalDifferentiation<F> {
    fn value(&self, position: &[f64]) -> f64 {
        self.function.value(position)
    }
}

impl<F: Function> Derivative1 for NumericalDifferentiation<F> {
    fn gradient(&self, position: &[f64]) -> Vec<f64> {
        let mut x: Vec<_> = position.iter().cloned().collect();

        let current = self.value(&x);

        position.iter().cloned().enumerate().map(|(i, x_i)| {
            let h = if x_i == 0.0 {
                EPSILON * 1.0e10
            } else {
                (EPSILON * x_i.abs()).sqrt()
            };

            assert!(h.is_finite());

            x[i] = x_i + h;

            let forward = self.function.value(&x);

            x[i] = x_i;

            let d_i = (forward - current) / h;

            assert!(d_i.is_finite());

            d_i
        }).collect()
    }
}

impl<F: Function + Default> Default for NumericalDifferentiation<F> {
    fn default() -> Self {
        NumericalDifferentiation::new(F::default())
    }
}

impl<F: Problem> Problem for NumericalDifferentiation<F> {
    fn dimensions(&self) -> usize {
        self.function.dimensions()
    }

    fn domain(&self) -> Vec<(f64, f64)> {
        self.function.domain()
    }

    fn minimum(&self) -> (Vec<f64>, f64) {
        self.function.minimum()
    }

    fn random_start(&self) -> Vec<f64> {
        self.function.random_start()
    }
}


#[cfg(test)]
mod tests {
    use types::Derivative1;
    use problems::{Problem, Sphere, Rosenbrock};
    use utils::are_close;
    use gd::GradientDescent;

    use super::NumericalDifferentiation;

    #[test]
    fn test_accuracy() {
        //let a = Sphere::default();
        let b = Rosenbrock::default();

        // FIXME: How to iterate over different problems?
        let problems = vec![b];

        for analytical_problem in problems {
            let numerical_problem = NumericalDifferentiation::new(analytical_problem);

            for _ in 0..1000 {
                let position = analytical_problem.random_start();

                let analytical_gradient = analytical_problem.gradient(&position);
                let numerical_gradient = numerical_problem.gradient(&position);

                assert_eq!(analytical_gradient.len(), numerical_gradient.len());

                assert!(analytical_gradient.into_iter().zip(numerical_gradient).all(|(a, n)|
                    a.is_finite() && n.is_finite() && are_close(a, n, 1.0e-3)
                ));
            }
        }
    }

    test_minimizer!{GradientDescent::new(),
        test_gd_sphere => NumericalDifferentiation::new(Sphere::default()),
        test_gd_rosenbrock => NumericalDifferentiation::new(Rosenbrock::default())}
}
