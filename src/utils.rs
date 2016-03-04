/// Tests whether we reached a flat area, i.e., tests if all absolute gradient component
/// lie within the `tolerance`.
pub fn is_saddle_point(gradient: &[f64], tolerance: f64) -> bool {
    gradient.iter().all(|dx| dx.abs() <= tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_saddle_point() {
        assert!(is_saddle_point(&[1.0, 2.0], 2.0));
        assert!(is_saddle_point(&[1.0, -2.0], 2.0));
        assert!(!is_saddle_point(&[1.0, 2.1], 2.0));
        assert!(!is_saddle_point(&[1.0, -2.1], 2.0));
    }
}
