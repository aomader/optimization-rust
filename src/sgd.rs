use log::LogLevel::Trace;
use rand::{SeedableRng, Rng, XorShiftRng, random};

use types::{Minimizer, Solution, Summation1};


/// Provides _stochastic_ Gradient Descent optimization.
pub struct StochasticGradientDescent {
    rng: XorShiftRng,
    max_iterations: Option<u64>,
    mini_batch: usize,
    step_width: f64
}

impl StochasticGradientDescent {
    /// Creates a new `StochasticGradientDescent` optimizer using the following defaults:
    ///
    /// - **`step_width`** = `0.01`
    /// - **`mini_batch`** = `1`
    /// - **`max_iterations`** = `1000`
    ///
    /// The used random number generator is randomly seeded.
    pub fn new() -> StochasticGradientDescent {
        StochasticGradientDescent {
            rng: random(),
            max_iterations: None,
            mini_batch: 1,
            step_width: 0.01
        }
    }

    /// Seeds the random number generator using the supplied `seed`.
    ///
    /// This is useful to create re-producable results.
    pub fn seed(&mut self, seed: [u32; 4]) -> &mut Self {
        self.rng = XorShiftRng::from_seed(seed);
        self
    }

    /// Adjusts the number of maximally run iterations. A value of `None` instructs the
    /// optimizer to ignore the nubmer of iterations.
    pub fn max_iterations(&mut self, max_iterations: Option<u64>) -> &mut Self {
        assert!(max_iterations.map_or(true, |max_iterations| max_iterations > 0));

        self.max_iterations = max_iterations;
        self
    }

    /// Adjusts the mini batch size, i.e., how many terms are considered in one step at most.
    pub fn mini_batch(&mut self, mini_batch: usize) -> &mut Self {
        assert!(mini_batch > 0);

        self.mini_batch = mini_batch;
        self
    }

    /// Adjusts the step size applied for each mini batch.
    pub fn step_width(&mut self, step_width: f64) -> &mut Self {
        assert!(step_width > 0.0);

        self.step_width = step_width;
        self
    }
}

impl<F: Summation1> Minimizer<F> for StochasticGradientDescent {
    type Solution = Solution;

    fn minimize(&self, function: &F, initial_position: Vec<f64>) -> Solution {
        let mut position = initial_position;
        let mut value = function.value(&position);

        if log_enabled!(Trace) {
            info!("Starting with y = {:?} for x = {:?}", value, position);
        } else {
            info!("Starting with y = {:?}", value);
        }

        let mut iteration = 0;
        let mut terms: Vec<_> = (0..function.terms()).collect();
        let mut rng = self.rng.clone();

        loop {
            // ensure that we don't run into cycles
            rng.shuffle(&mut terms);

            for batch in terms.chunks(self.mini_batch) {
                let gradient = function.partial_gradient(&position, batch);

                // step into the direction of the negative gradient
                for (x, g) in position.iter_mut().zip(gradient) {
                    *x -= self.step_width * g;
                }
            }

            value = function.value(&position);

            iteration += 1;

            if log_enabled!(Trace) {
                debug!("Iteration {:6}: y = {:?}, x = {:?}", iteration, value, position);
            } else {
                debug!("Iteration {:6}: y = {:?}", iteration, value);
            }

            let reached_max_iterations = self.max_iterations.map_or(false,
                |max_iterations| iteration == max_iterations);

            if reached_max_iterations {
                info!("Reached maximal number of iterations, stopping optimization");
                return Solution::new(position, value);
            }
        }
    }
}
