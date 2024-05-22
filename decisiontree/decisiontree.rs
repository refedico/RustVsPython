use ndarray_rand::rand::SeedableRng;
use rand::rngs::SmallRng;
use memory_stats::memory_stats;

use linfa::prelude::*;
use linfa_trees::{DecisionTree, Result, SplitQuality};

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

fn main() -> Result<()> {
    // Monitoraggio memoria prima del caricamento del dataset
    print_memory_usage("Before loading dataset");

    // Carica il dataset Iris
    let mut rng = SmallRng::seed_from_u64(42);

    let (train, test) = linfa_datasets::iris()
        .shuffle(&mut rng)
        .split_with_ratio(0.8);

    // Monitoraggio memoria dopo il caricamento del dataset
    print_memory_usage("After loading dataset");

    println!("Training model with Gini criterion ...");
    print_memory_usage("Before training Gini model");

    let gini_model = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .max_depth(Some(100))
        .min_weight_split(1.0)
        .min_weight_leaf(1.0)
        .fit(&train)?;

    // Monitoraggio memoria dopo l'addestramento del modello Gini
    print_memory_usage("After training Gini model");

    let gini_pred_y = gini_model.predict(&test);
    let cm = gini_pred_y.confusion_matrix(&test)?;

    println!("{:?}", cm);

    println!(
        "Test accuracy with Gini criterion: {:.2}%",
        100.0 * cm.accuracy()
    );

    let feats = gini_model.features();
    println!("Features trained in this tree {:?}", feats);

    Ok(())
}
