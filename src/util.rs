use rand::Rng;

pub fn get_mean(vec: Vec<f64>) -> f64 {
    vec.iter().sum::<f64>() / vec.len() as f64
}

pub fn get_rand() -> f64 {
    let mut rng = rand::thread_rng();

    rng.gen::<f64>();
}