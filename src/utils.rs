#[cfg(test)]
use std::f64::{MIN_POSITIVE, MAX};


/// Tests whether we reached a flat area, i.e., tests if all absolute gradient component
/// lie within the `tolerance`.
pub fn is_saddle_point(gradient: &[f64], tolerance: f64) -> bool {
    gradient.iter().all(|dx| dx.abs() <= tolerance)
}


/// Tests whether two floating point numbers are close using the relative error
/// and handling special cases like infinity etc.
#[cfg(test)]
pub fn are_close(a: f64, b: f64, eps: f64) -> bool {
    assert!(eps.is_finite());

    let d = (a - b).abs();

    // identical, e.g., infinity
    a == b

    // a or b is zero or both are extremely close to it
    // relative error is less meaningful here
    || ((a == 0.0 || b == 0.0 || d < MIN_POSITIVE) &&
        d < eps * MIN_POSITIVE)

    // finally, use the relative error
    || d / (a + b).min(MAX) < eps
}


#[cfg(test)]
mod tests {
    use std::f64::{INFINITY, NAN};

    use super::{is_saddle_point, are_close};

    #[test]
    fn test_is_saddle_point() {
        assert!(is_saddle_point(&[1.0, 2.0], 2.0));
        assert!(is_saddle_point(&[1.0, -2.0], 2.0));
        assert!(!is_saddle_point(&[1.0, 2.1], 2.0));
        assert!(!is_saddle_point(&[1.0, -2.1], 2.0));
    }

    #[test]
    fn test_are_close() {
        assert!(are_close(1.0, 1.0, 0.00001));
        assert!(are_close(INFINITY, INFINITY, 0.00001));
        assert!(are_close(1.0e-1000, 0.0, 0.1));
        assert!(!are_close(1.0e-40, 0.0, 0.000001));
        assert!(!are_close(2.0, 1.0, 0.00001));
        assert!(!are_close(NAN, NAN, 0.00001));
    }
}
