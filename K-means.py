import numpy as np

def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    return data

def initialize_centroids(data, k):
    return data[np.random.choice(data.shape[0], k, replace=False)]

def assign_clusters(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(data, clusters, k):
    new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(k)])
    return new_centroids

def kmeans(data, k, max_iters=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters

def save_clusters(file_path, clusters):
    with open(file_path, 'w') as f:
        for location_id, cluster_label in enumerate(clusters):
            f.write(f"{location_id} {cluster_label}\n")

# Main execution
if __name__ == "__main__":
    input_file = "C:\\Users\HP\OneDrive\Desktop\place.txt"  # Path to the input file
    output_file = "C:\\Users\HP\OneDrive\Desktop\cluster.txt"  # Path to the output file
    k = 3  # Number of clusters

    data = load_data(input_file)
    clusters = kmeans(data, k)
    save_clusters(output_file, clusters)