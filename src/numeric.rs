use std::f64::EPSILON;

use problems::Problem;
use types::{Function, DifferentiableFunction};


/// Wraps a function for which to provide numeric differentiation.
///
/// Uses simple one step forward finite difference with step width `h = √εx`.
pub struct Numerical<F: Function> {
    function: F
}

impl<F: Function> Numerical<F> {
    pub fn new(function: F) -> Self {
        Numerical {
            function: function
        }
    }
}

impl<F: Function> Function for Numerical<F> {
    fn value(&self, position: &[f64]) -> f64 {
        self.function.value(position)
    }
}

impl<F: Function> DifferentiableFunction for Numerical<F> {
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

            let dx_i = (forward - current) / h;

            assert!(dx_i.is_finite());

            dx_i
        }).collect()
    }
}

impl<F: Function + Default> Default for Numerical<F> {
    fn default() -> Self {
        Numerical::new(F::default())
    }
}

impl<F: Problem> Problem for Numerical<F> {
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
    use types::DifferentiableFunction;
    use problems::{Problem, Sphere, Rosenbrock};
    use utils::are_close;
    use gd::GradientDescent;

    use super::Numerical;

    #[test]
    fn test_accuracy() {
        //let a = Sphere::default();
        let b = Rosenbrock::default();

        // FIXME: How to iterate over different problems?
        let problems = vec![b];

        for analytical_problem in problems {
            let numerical_problem = Numerical::new(analytical_problem.clone());

            for _ in 0..1000 {
                let position = analytical_problem.random_start();

                let analytical_gradient = analytical_problem.gradient(&position);
                let numerical_gradient = numerical_problem.gradient(&position);

                assert_eq!(analytical_gradient.len(), numerical_gradient.len());

                assert!(analytical_gradient.into_iter().zip(numerical_gradient).all(|(a, n)| {
                    a.is_finite() && n.is_finite() && are_close(a, n, 1.0e-3)
                }));
            }
        }
    }

    test_minimizer!{GradientDescent::new(),
        test_gd_sphere => Numerical::new(Sphere::default()),
        test_gd_rosenbrock => Numerical::new(Rosenbrock::default())}
}
