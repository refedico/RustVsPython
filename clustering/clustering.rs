use linfa::traits::Fit;
use linfa::traits::Predict;
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use linfa_datasets::generate;
use linfa_nn::distance::LInfDist;
use memory_stats::memory_stats;
use ndarray::{array, Axis};
use ndarray_npy::write_npy;
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use std::time::{Duration, SystemTime};

fn log_memory_usage(message: &str) {
    if let Some(usage) = memory_stats() {
        println!("{}: Memory used: {} KB", message, usage.physical_mem/1024);
    } else {
        println!("{}: Couldn't retrieve memory usage.", message);
    }
}

// A routine K-means task: build a synthetic dataset, fit the algorithm on it
// and save both training data and predictions to disk.
fn main() {
    log_memory_usage("Before dataset generation");

    // Our random number generator, seeded for reproducibility
    let mut rng = Xoshiro256Plus::seed_from_u64(42);

    // For each our expected centroids, generate n data points around it (a "blob")
    let expected_centroids = array![[10., 10.], [1., 12.], [20., 30.], [-20., 30.]];
    let n = 10000;
    let dataset = DatasetBase::from(generate::blobs(n, &expected_centroids, &mut rng));

    log_memory_usage("After dataset generation");

    let now = SystemTime::now();

    // Configure our training algorithm
    let n_clusters = expected_centroids.len_of(Axis(0));
    let model = KMeans::params_with(n_clusters, rng, LInfDist)
        .max_n_iterations(200)
        .tolerance(1e-5)
        .fit(&dataset)
        .expect("KMeans fitted");

    log_memory_usage("After model fitting");

    // Assign each point to a cluster using the set of centroids found using fit
    let dataset = model.predict(dataset);
    let DatasetBase {
        records, targets, ..
    } = dataset;

    let elapsed = now.elapsed().expect("Time went backwards");
    println!("Elapsed time: {:?}", elapsed);

    log_memory_usage("After prediction");

    // Save to disk our dataset (and the cluster label assigned to each observation),
    // we use the npy format for compatibility with NumPy
    write_npy("clustered_dataset.npy", &records).expect("Failed to write .npy file");
    write_npy("clustered_memberships.npy", &targets.map(|&x| x as u64))
    .expect("Failed to write .npy file");

    log_memory_usage("After saving to disk");
}
