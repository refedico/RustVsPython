import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import psutil
import os

def log_memory_usage(message):
    process = psutil.Process(os.getpid())
    print(f"{message}: Memory used: {process.memory_info().rss} bytes")

# A routine K-means task: build a synthetic dataset, fit the algorithm on it
# and save both training data and predictions to disk.
def main():
    log_memory_usage("Before dataset generation")

    # Our random number generator, seeded for reproducibility
    rng = np.random.RandomState(42)

    # For each our expected centroids, generate n data points around it (a "blob")
    expected_centroids = np.array([[10., 10.], [1., 12.], [20., 30.], [-20., 30.]])
    n = 10000
    X, _ = make_blobs(n_samples=n * len(expected_centroids), centers=expected_centroids, random_state=rng)

    log_memory_usage("After dataset generation")

    # Configure our training algorithm
    n_clusters = expected_centroids.shape[0]
    kmeans = KMeans(n_clusters=n_clusters, max_iter=200, tol=1e-5, random_state=rng)
    kmeans.fit(X)

    log_memory_usage("After model fitting")

    # Assign each point to a cluster using the set of centroids found using fit
    labels = kmeans.predict(X)

    log_memory_usage("After prediction")

    # Save to disk our dataset (and the cluster label assigned to each observation)
    # We use the npy format for compatibility with NumPy
    # np.save("clustered_dataset.npy", X)
    # np.save("clustered_memberships.npy", labels)

    # log_memory_usage("After saving to disk")

main()
