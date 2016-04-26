use std::borrow::Borrow;
use std::ops::Deref;


/// Defines an objective function `f` that is subject to minimization.
///
/// For convenience every function with the same signature as `value()` qualifies as
/// an objective function, e.g., minimizing a closure is perfectly fine.
pub trait Function {
    /// Computes the objective function at a given `position` `x`, i.e., `f(x) = y`.
    fn value(&self, position: &[f64]) -> f64;
}

impl<F: Fn(&[f64]) -> f64> Function for F {
    fn value(&self, position: &[f64]) -> f64 {
        self(position)
    }
}


/// Defines an objective function `f` that is able to compute the first derivative
/// `f'(x)` analytically.
pub trait Derivative1: Function {
    /// Computes the gradient of the objective function at a given `position` `x`,
    /// i.e., `∀ᵢ ∂/∂xᵢ f(x) = ∇f(x)`.
    fn gradient(&self, position: &[f64]) -> Vec<f64>;
}


/// Represents a summation of different functions.
pub trait SummationFunction: Function {
    /// Number of terms of the summation.
    fn terms(&self) -> usize;

    /// Computes the partial value over a set of `terms` at the given `position`.
    fn partial_value<I: Borrow<usize>, T: IntoIterator<Item=I>>(&self, position: &[f64], terms: T) -> f64;
}


/// Represents a summation of function that support the computation of the first derivative.
pub trait SummationDerivative1: SummationFunction + Derivative1 {
    /// Computes the partial gradient over a set of `terms` at the given `position`.
    fn partial_gradient<T: IntoIterator<Item=usize>>(&self, position: &[f64], terms: T) -> Vec<f64>;
}


/// New-type to support summation over common collection types.
pub struct Summation<T>(T);

impl<C: Deref<Target=[F]>, F: Function> Function for Summation<C> {
    fn value(&self, position: &[f64]) -> f64 {
        self.partial_value(position, 0..self.0.len())
    }
}

impl<C: Deref<Target=[F]>, F: Function> SummationFunction for Summation<C> {
    fn terms(&self) -> usize {
        self.0.len()
    }

    fn partial_value<I: Borrow<usize>, T: IntoIterator<Item=I>>(&self, position: &[f64], terms: T) -> f64 {
        let mut value = 0.0;

        for term in terms {
            value += self.0[*term.borrow()].value(position);
        }

        value
    }
}

impl<C: Deref<Target=[F]>, F: Derivative1> Derivative1 for Summation<C> {
    fn gradient(&self, position: &[f64]) -> Vec<f64> {
        self.partial_gradient(position, 0..self.0.len())
    }
}

impl<C: Deref<Target=[F]>, F: Derivative1> SummationDerivative1 for Summation<C> {
    fn partial_gradient<T: IntoIterator<Item=usize>>(&self, position: &[f64], terms: T) -> Vec<f64>
    {
        let mut gradient = vec![0.0; position.len()];

        // TODO: This can be optimized easily
        for term in terms.into_iter() {
            for (g, gi) in gradient.iter_mut().zip(self.0[term].gradient(position)) {
                *g += gi;
            }
        }

        gradient
    }
}


pub struct Foo;

impl Function for Foo {
    fn value(&self, _: &[f64]) -> f64 {
        0.0
    }
}

pub fn test() {
    let s = Summation(vec![Foo, Foo]);
    let asd = vec![Foo, Foo, Foo];
    let s2 = Summation(&*asd);
}


/*
pub trait FunctionSummation: Summation
    where Self::Term: Function
{
}

pub trait Derivative1Summation: Summation
    where Self::Term: Derivative1
{
    fn partial_gradient(&self, position: &[f64], terms: &[usize]) -> Vec<f64> {
        vec![]
        let mut value = 0.0;

        let functions = self.terms();

        for &term in terms {
            value += functions[term].value(position);
        }

        value
    }
}

impl<T> Summation for Vec<T> {
    type Term = T;

    fn terms(&self) -> &[T] {
        &*self
    }
}

impl<T: Function> FunctionSummation for Vec<T> {}
impl<T: > FunctionSummation for Vec<T> {}
*/

/*
pub trait Summation {
    type Term: Function;

    fn terms(&self) -> &[Self::Term];

    fn partial_value(&self, position: &[f64], terms: &[usize]) -> f64 {
        let mut value = 0.0;

        let functions = self.terms();

        for &term in terms {
            value += functions[term].value(position);
        }

        value
    }
}

pub trait SummationDerivative1: Summation{
    fn partial_gradient(&self, terms: &[usize]) -> Vec<f64> {
        vec![]
    }
}

*/


/*
impl<F: Function> Function for Summation<F> {
    fn value(&self, position: &[f64]) -> f64 {
        let mut value = 0.0;

        for function in self.terms() {
            value += function.value(position);
        }

        value
    }
}
*/

/*
pub trait SummationFunction<F: Function> {
    fn terms(&self) -> &[F];

    fn partial_value(&self, &) -> f64 {
        let mut value = 0.0;

        for term in self.terms() {
            value += term.value();
        }

        value
    }
}
*/
/*
pub trait SummationDerivative1<F: Derivative1>: SummationFunction<F> {

}
*/

/*
pub trait Summation2 {
    type Term: Function;

    fn terms(&self) -> &[Self::Term];

    fn partial_value(&self) ->
}

impl<S: Summation2> S where S::Term: Derivative1 {
    fn asd() {}
}
*/


/*
impl<S: Summation2> Function for S
    where S::Term: Function
{
    fn value(&self, position: &[f64]) -> f64 {
        let mut value = 0.0;

        for function in self.terms() {
            value += function.value(position);
        }

        value
    }
}
*/


/// Defines an optimizer that is able to minimize a given objective function `F`.
pub trait Minimizer<F: ?Sized> {
    type Solution: Evaluation;

    /// Performs the actual minimization and returns a solution that
    /// might be better than the initially provided one.
    fn minimize(&self, function: &F, initial_position: Vec<f64>) -> Self::Solution;
}


/// Captures the essence of a function evaluation.
pub trait Evaluation {
    /// Position `x` with the lowest corresponding value `f(x)`.
    fn position(&self) -> &[f64];

    /// The actual value `f(x)`.
    fn value(&self) -> f64;
}


/// A solution of a minimization run providing only the minimal information.
///
/// Each `Minimizer` might yield different types of solution structs which provide more
/// information.
#[derive(Debug, Clone)]
pub struct Solution {
    /// Position `x` of the lowest corresponding value `f(x)` that has been found.
    pub position: Vec<f64>,
    /// The actual value `f(x)`.
    pub value: f64
}

impl Solution {
    /// Creates a new `Solution` given the `position` as well as the corresponding `value`.
    pub fn new(position: Vec<f64>, value: f64) -> Solution {
        Solution {
            position: position,
            value: value
        }
    }
}

impl Evaluation for Solution {
    fn position(&self) -> &[f64] {
        &self.position
    }

    fn value(&self) -> f64 {
        self.value
    }
}
