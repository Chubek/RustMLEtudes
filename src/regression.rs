mod util;
use util::get_mean;

struct LinReg {
    X: Vec<f64>,
    y: Vec<f64>,
    test_ratio: f64
}

trait LeastSquares {
    fn new(X: Vec<f64>, y: Vec<f64>, test_ratio: f64) -> Self;
    fn test_split(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
    fn get_X_mean(&self) -> Option<f64>;
    fn get_Y_mean(&self) -> Option<f64>;
    fn calculate_diff(&self, X_mean: f64, y_mean: f64) -> Option<f64>;
    fn calculate_intercept(X_mean: f64, y_mean: f64, diff: f64) -> Option<f64>;
    fn calculate_rmse(y_test: Vec<f64>, y_pred: Vec<f64>) -> Option<f64>;
    fn fit(&self) -> (f64, f64, f64);
}

impl LeastSquares for LinReg {

    fn new(X: Vec<f64>, y: Vec<f64>, test_ratio: f64) -> LinReg {
        LinReg {X: X, y: y, test_ratio: test_ratio}
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

    fn calculate_diff(&self, X_mean: f64, y_mean: f64) -> Option<f64> {

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

    fn calculate_intercept(&self, X_mean: f64, y_mean: f64, diff: f64) -> Option<f64> {
        
        match (X_mean == 0 || y_mean == 0) {
            true => None,
            _ => Some(y_mean - (diff * X_mean)),
        }
    }

    fn calculate_rmse(y_test: Vec<f64>, y_pred: Vec<f64>) -> Option<f64> {

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


