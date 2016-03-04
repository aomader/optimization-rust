use rand::{Rng, SeedableRng, XorShiftRng};

use types::*;
use line_search::*;
use utils::flat;

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

impl<T: SampledDifferentiableObjective> Optimizer<T> for StochasticGradientDescent {
    fn optimize(&self, objective: &T, x0: Vec<f64>) -> Solution {
        let mut samples: Vec<_> = (0..objective.samples()).collect();

        let mut rng = XorShiftRng::from_seed([1337, 42, 99999, 314]);

        //rng.shuffle(&mut samples);

        //let mut train: Vec<_> = samples.iter().cloned().take(samples.len() / 2).collect();
        //let test: Vec<_> = samples.iter().cloned().skip(samples.len() / 2).collect();


        let mut x = x0;
        let mut y = objective.value(&x);

        let line_search = NoLineSearch::new(1.0e-3);

        let mut iteration = 0;

        loop {

            //rng.shuffle(&mut train);
            rng.shuffle(&mut samples);

            for batch in samples.chunks(self.mini_batch) {
                let mini_batch = MiniBatchObjective {
                    objective: objective,
                    samples: batch
                };

                let gradient = mini_batch.gradient(&x);

                if flat(&gradient, 1.0e-8) {
                    info!("Gradient to small, stopping optimization");

                    return Solution {
                        x: x,
                        y: y
                    }
                }

                let direction: Vec<_> = gradient.into_iter().map(|g| -g).collect();

                let (line_x, line_y) = line_search.search(&mini_batch, &x, &direction);

                x = line_x;
                y = line_y;

                //trace!("Next batch y = {:e}", y);
            }

            iteration += 1;

/*
                    let train_batch = MiniBatchObjective {
                        objective: objective,
                        samples: &train
                    };
                    let test_batch = MiniBatchObjective {
                        objective: objective,
                        samples: &test
                    };*/

            y = objective.value(&x);
            //y = train_batch.value(&x);
            //let val = test_batch.value(&x);

            //debug!("Iteration {:4}: y = {:?} ; val = {:?}", iteration, y, val);
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

struct MiniBatchObjective<'a, T: 'a> {
    objective: &'a T,
    samples: &'a [usize]
}

impl<'a, T: SampledObjective> Objective for MiniBatchObjective<'a, T> {
    fn value(&self, x: &[f64]) -> f64 {
        self.samples.iter().map(|&i| self.objective.sample_value(x, i)).sum()
    }
}

impl<'a, T: SampledDifferentiableObjective> DifferentiableObjective for MiniBatchObjective<'a, T> {
    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        let mut gradient = vec![0.0; x.len()];
        for &i in self.samples {
            for (g, gi) in gradient.iter_mut().zip(self.objective.sample_gradient(x, i)) {
                *g += gi;
            }
        }
        gradient
    }
}
