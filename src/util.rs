pub fn get_mean(vec: Vec<f64>) -> f64 {
    vec.iter().sum::<f64>() / vec.len() as f64
}
