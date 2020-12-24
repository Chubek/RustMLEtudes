mod util;
use util::get_mean;
use util::get_rand;

struct Reg {
    X: Vec<f64>,
    y: Vec<f64>,
    test_ratio: f64
}

trait LeastSquares {
    fn new(X: Vec<f64>, y: Vec<f64>, test_ratio: f64) -> Self;
    fn test_split(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
    fn get_X_mean(&self) -> Option<f64>;
    fn get_Y_mean(&self) -> Option<f64>;
    fn calculate_diff(&self, X_mean: &f64, y_mean: &f64) -> Option<f64>;
    fn calculate_intercept(X_mean: &f64, y_mean: f64, diff: &f64) -> Option<f64>;
    fn calculate_rmse(y_test: &Vec<f64>, y_pred: &Vec<f64>) -> Option<f64>;
    fn fit(&self) -> (f64, f64, f64);
}

impl LeastSquares for Reg {

    fn new(X: Vec<f64>, y: Vec<f64>, test_ratio: f64) -> Reg {
        Reg {X: X, y: y, test_ratio: test_ratio}
    }

    fn test_split(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let len_appl = self.X.len() * self.test_ratio;

        X_train = &self.X[..len_appl].as_vec();
        X_test = &self.X[len_appl..].as_vec();

        y_train = &self.y[..len_appl].as_vec();
        y_test = &self.y[len_appl..].as_vec();

        (X_train, X_test, y_train, y_test)

    }

    fn get_X_mean(&self) -> Option<f64> {
        match self.X.len() {
            0 => None,
            _ => Some(get_mean(self.X)),
        }

    }

    fn get_y_mean(&self) -> Option<f64> {
        match self.y.len() {
            0 => None,
            _ => Some(get_mean(self.y)),
        }

    }

    fn calculate_diff(&self, X_mean: &f64, y_mean: &f64) -> Option<f64> {

        let num = self.X.iter()
                    .zip(self.y.iter())
                    .map(|(i, j)| (i - X_mean) * (j - y_mean))
                    .iter()
                    .sum();
        let denom = self.X.iter()
                    .map(|i| (i - X_mean)
                    .pow(2))
                    .iter()
                    .sum();

        match denom {
            0 => None,
            _ => Some(num / denom),
        }
    }

    fn calculate_intercept(&self, X_mean: &f64, y_mean: &f64, diff: &f64) -> Option<f64> {
        
        match (X_mean == 0 || y_mean == 0) {
            true => None,
            _ => Some(y_mean - (diff * X_mean)),
        }
    }

    fn calculate_rmse(y_test: &Vec<f64>, y_pred: &Vec<f64>) -> Option<f64> {

        match (y_test.len() == y_pred.len()) {
            false => None,
            _ => y_pred.iter().zip(y_test.iter())
                                .map(|(p, t)| (p - t)
                                .pow(2))
                                .iter()
                                .map(|rmse| (rmse / y_pred.len()))
                                .iter()
                                .sum()
                                .sqrt()


        }

    }

}


trait LogisticReg {
    fn new(X: Vec<f64>, y: Vec<f64>, test_ratio: f64) -> Self;
    fn test_split(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
    fn grad_descent(&self, x_in: &Vec<f64>, y_in: &Vec<f64>, learn_rate: &f64, init_weights: &Vec<f64>) -> Option<Vec<f64>>;
    fn sigmoid(a: f64) -> Option<f64>;
    fn stochastic_grad_descent(&self, x_in: &Vec<f64>, y_in: &Vec<f64>, max_cycles: i32, init_weights: &Vec<f64>) -> Option<Vec<f64>>;
    fn classify(&self, x_in: &Vec<f64>, weights: &Vec<i32>) -> Option<f64>;
    fn error_rate(y_pred: &Vec<f64>, y_test: &Vec<f64>) ->  Option<f64>;
    fn train(&self);
}

impl LogisticReg for Reg {
    fn new(X: Vec<f64>, y: Vec<f64>, test_ratio: f64) -> Reg {
        Reg {X: X, y: y, test_ratio: test_ratio}
    }

    fn test_split(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let len_appl = self.X.len() * self.test_ratio;

        X_train = &self.X[..len_appl].as_vec();
        X_test = &self.X[len_appl..].as_vec();

        y_train = &self.y[..len_appl].as_vec();
        y_test = &self.y[len_appl..].as_vec();

        (X_train, X_test, y_train, y_test)

    }

    fn sigmoid(a: f64) {
        1.0 / (1.0 + (-1 * a).exp())
    }

    fn grad_descent(&self, x_in: &Vec<f64>, y_in: &Vec<f64>, learn_rate: &f64, init_weights: &Vec<i32>) -> Option<Vec<i32>> {
        if x_in.len() < 1 || init_weights.len() != x_in.len() || x_in.len() != y_in.len() {
            None
        }

        let weights = init_weights.clone();
        let alpha = learn_rate.clone()

        for i in 0..x_in.len() {
            let alpha = 4 / (1.0 + i) + alpha;
            let logit_res = weights
                                .iter()
                                .zip(x_in.iter())
                                .map(|(x, y)| self.sigmoid(x * y)?)
                                .collect();
            let error = y_in.iter()
                                .zip(logit_res.iter())
                                .map(|(x, y)| x  - y)
                                .collect();
            let descent = x_in.iter()
                                .zip(error.iter())
                                .map(|(x, y)| alpha * x * y)
                                .collect()
            let weights = weights.iter()
                                .zip(descent.iter())
                                .map(|x, y| x - y)
                                .collect();
        }        

        Some(weights)
    }

    fn stochastic_grad_descent(&self, x_in: &Vec<f64>, y_in: &Vec<f64>, max_cycles: i32, init_weights: &Vec<f64>) -> Option<Vec<f64>> {
        if x_in.len() < 1 || init_weights.len() != x_in.len() || x_in.len() != y_in.len() {
            None
        }

        let weights: Vec<f64> = vec![1.00, x_in.len()];

        for _ in 0..max_cycles {
            let alpha = get_rand();

            weights = self.grad_descent(x_in, y_in, &alpha, &weights)?;

        }

        Some(weights)


    }

    fn classify(&self, x_in: &Vec<f64>, weights: &Vec<i32>) -> Option<f64> {

        if x_in.len() != weights.len() {
            None
        }

        let prob = x_in.iter()
                            .zip(weights.iter())
                            .map(|(x, y)| self.sigmoid(x  y))
                            .iter()
                            .sum();

        match (prob > 0.5) {
            true => Some(1.0),
            false => Some(0.0),
        }

    }

    fn error_rate(y_pred: &Vec<f64>, y_test: &Vec<f64>) -> Option<f64> {

        if y_pred.len() != y_test.len() {
            None
        }

        let error_delta = y_pred.iter().zip(y_test.iter()).map(|(x, y)| x == y).collect();

        let mut i = 0;
        for error in error_delta {
            if error {
                i += 1;
            }
        }

        i / y_test.len()

    }


}