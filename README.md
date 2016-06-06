# optimization [![Build Status](https://travis-ci.org/b52/optimization-rust.svg?branch=master)](https://travis-ci.org/b52/optimization-rust) [![Coverage Status](https://coveralls.io/repos/b52/optimization-rust/badge.svg?branch=master&service=github)](https://coveralls.io/github/b52/optimization-rust?branch=master) [![crates.io version](http://meritbadge.herokuapp.com/optimization)](https://crates.io/crates/optimization)

Collection of optimization algorithms and strategies.

## Usage

```rust
extern crate optimization;

use optimmization::{Minimizer, GradientDescent, NumericalDifferentiation, Func};

// numeric version of the Rosenbrock function
let function = NumericalDifferentiation::new(Func(|x: &[f64]| {
    (1.0 - x[0]).powi(2) + 100.0*(x[1] - x[0].powi(2)).powi(2)
}));

// we use a simple gradient descent scheme
let minimizer = GradientDescent::new();

// perform the actual minimization, depending on the task this may
// take some time, it may be useful to install a log sink to see
// what's going on
let solution = minimizer.minimize(&function, vec![-3.0, -4.0]);

println!("Found solution for Rosenbrock function at f({:?}) = {:?}",
    solution.position, solution.value);
```

## Installation

Simply add it as a `Cargo` dependency:

```toml
[dependencies]
optimization = "*"
```

## Documentation

For an exhaustive documentation head over to the [API docs].

## Development

In order to run the code against more sophisticated linters compile it
using a nightly version in combination with the feature `unstable`:

```shell
$ cargo test --features unstable
```

## License

This software is licensed under the terms of the MIT license. Please see the
[LICENSE](LICENSE) for full details.

[API docs]: https://b52.github.io/optimization-rust
