use rand::{Rng, SeedableRng, XorShiftRng};
use std::ops::Deref;

use types::*;
use line_search::*;
use utils::is_saddle_point;


pub struct StochasticGradientDescent {
    mini_batch: usize,
    max_iterations: Option<u64>
}

impl StochasticGradientDescent {
    pub fn new() -> StochasticGradientDescent {
        StochasticGradientDescent {
            mini_batch: 1,
            max_iterations: None
        }
    }

    pub fn mini_batch(mut self, mini_batch: usize) -> Self {
        assert!(mini_batch > 0);

        self.mini_batch = mini_batch;
        self
    }

    pub fn max_iterations(mut self, max_iterations: Option<u64>) -> Self {
        assert!(max_iterations.map_or(true, |max_iterations| max_iterations > 0));

        self.max_iterations = max_iterations;
        self
    }
}

impl<'a, C, F> Optimizer<Summation<C>> for StochasticGradientDescent
    where C: Deref<Target=[F]>,
          F: DifferentiableFunction
{
    fn optimize(&self, summation: &Summation<C>, x0: Vec<f64>) -> Solution {
        let mut objectives: Vec<_> = summation.terms().iter().collect();

        let mut rng = XorShiftRng::from_seed([1337, 42, 99999, 314]);

        let line_search = ArmijoLineSearch::new(0.1, 1.0, 0.9);

        let mut x = x0;
        let mut y = summation.value(&x);

        let mut iteration = 0;

        loop {
            // ensure that we don't run into cycles
            rng.shuffle(&mut objectives);

            for batch in objectives.chunks(self.mini_batch) {
                let batch = Summation::new(batch);

                let gradient = batch.gradient(&x);

                let direction: Vec<_> = gradient.into_iter().map(|g| -g).collect();

                let line_x = line_search.search(&batch, &x, &direction);

                x = line_x;
            }

            y = summation.value(&x);
            iteration += 1;

            debug!("Iteration {:4}: y = {:?}", iteration, y);

            let reached_max_iterations = self.max_iterations.map_or(false,
                                                                    |max_iterations| iteration == max_iterations);

            if reached_max_iterations {
                info!("Reached maximal number of iterations, stopping optimization");

                return Solution {
                    x: x,
                    y: y
                }
            }
        }
    }
}
