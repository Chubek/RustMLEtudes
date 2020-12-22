mod util;
use util::get_mean;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(crate::get_mean(vec![2.0, 2.0]), 2.0);
    }
}
