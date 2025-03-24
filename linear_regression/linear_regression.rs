use linfa::prelude::*;
use linfa_elasticnet::{ElasticNet, Result};
use ndarray::Array1;
use memory_stats::memory_stats;

fn print_memory_usage(message: &str) {
    if let Some(usage) = memory_stats() {
        println!(
            "{} - RSS: {:.2} MB",
            message,
            usage.physical_mem as f64 / (1024.0 * 1024.0)
        );
    } else {
        println!("Couldn't retrieve memory usage information.");
    }
}

fn mean_squared_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let diff = y_true - y_pred;
    diff.mapv(|x| x.powi(2)).mean().unwrap()
}

fn main() -> Result<()> {
    // Measuring memory usage before loading the dataset
    print_memory_usage("Before loading dataset");

    let (train, valid) = linfa_datasets::diabetes().split_with_ratio(0.90);

    // Measuring memory usage after dataset loading
    print_memory_usage("After loading dataset");

    // Train a LASSO model with a penalty of 0.3
    print_memory_usage("Before training the model");
    let model = ElasticNet::params()
        .penalty(0.3)
        .l1_ratio(1.0)
        .fit(&train)?;
    print_memory_usage("After training the model");

    // println!("z score: {:?}", model.z_score());

    // Calculate R² on the validation set
    print_memory_usage("Before prediction");
    let y_est = model.predict(&valid);
    print_memory_usage("After prediction");
    println!("predicted variance: {}", valid.r2(&y_est)?);

    // Calculate the mean square error (MSE)
    let mse = mean_squared_error(&valid.targets, &y_est);
    println!("Mean Squared Error: {}", mse);

    Ok(())
}
