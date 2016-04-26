use line_search::{LineSearch, FixedStepWidth};
use types::{Minimizer, Derivative1, Solution, SummationDerivative1};


/// Provides _stochastic_ Gradient Descent optimization.
pub struct StochasticGradientDescent<L> {
    line_search: L,
    gradient_tolerance: f64,
    max_iterations: Option<u64>,
    mini_batch: u64
}

impl StochasticGradientDescent<FixedStepWidth> {
    pub fn new() -> StochasticGradientDescent<FixedStepWidth> {
        StochasticGradientDescent {
            line_search: FixedStepWidth::new(0.01),
            gradient_tolerance: 1.0e-4,
            max_iterations: None,
            mini_batch: 1
        }
    }
}

impl<L> StochasticGradientDescent<L> {
    /// Specifies the line search method to use.
    pub fn line_search<S: LineSearch>(self, line_search: S) -> StochasticGradientDescent<S> {
        StochasticGradientDescent {
            line_search: line_search,
            gradient_tolerance: self.gradient_tolerance,
            max_iterations: self.max_iterations,
            mini_batch: self.mini_batch
        }
    }

    /// Adjusts the gradient tolerance which is used as abort criterion to decide
    /// whether we reached a plateau.
    pub fn gradient_tolerance(&mut self, gradient_tolerance: f64) -> &mut Self {
        assert!(gradient_tolerance > 0.0);

        self.gradient_tolerance = gradient_tolerance;
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
    pub fn mini_batch(&mut self, mini_batch: u64) -> &mut Self {
        assert!(mini_batch > 0);

        self.mini_batch = mini_batch;
        self
    }
}

impl<F: SummationDerivative1, L: LineSearch> Minimizer<F> for StochasticGradientDescent<L>
{
    type Solution = Solution;

    fn minimize(&self, function: &F, initial_position: Vec<f64>) -> Solution {
        let terms: Vec<_> = (0..function.terms()).collect();

        let pv = function.partial_value(&initial_position, &*terms); //terms.iter().cloned());

        return Solution::new(initial_position, 0.0);
    }
}

/*
impl<S: Summation2, L: LineSearch> Minimizer<S> for StochasticGradientDescent<L>
    where S::Term: Derivative1
{
    type Solution = Solution;

    fn minimize(&self, function: &F, initial_position: Vec<f64>) -> Solution {
        return Solution::new(initial_position, 0.0);

        info!("Starting gradient descent minimization: gradient_tolerance = {:?},
            max_iterations = {:?}, line_search = {:?}",
        self.gradient_tolerance, self.max_iterations, self.line_search);

        let mut position = initial_position;
        let mut value = function.value(&position);

        if log_enabled!(Trace) {
            info!("Starting with y = {:?} for x = {:?}", value, position);
        } else {
            info!("Starting with y = {:?}", value);
        }

        let mut iteration = 0;

        loop {
            let gradient = function.gradient(&position);

            if is_saddle_point(&gradient, self.gradient_tolerance) {
                info!("Gradient to small, stopping optimization");

                return Solution::new(position, value);
            }

            let direction: Vec<_> = gradient.into_iter().map(|g| -g).collect();

            let iter_xs = self.line_search.search(function, &position, &direction);

            position = iter_xs;
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
        */

