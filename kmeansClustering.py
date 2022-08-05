import numpy as np
import matplotlib.pyplot as plt


def centroids_initialization(data_points, k, opt="forgy"):
    data_points_len = data_points.shape[0]

    match opt:
        case "forgy":
            x = np.random.choice(data_points_len, k, replace=False)
            return data_points[x, :]

        case "random partition":
            indices = np.random.choice(range(k), replace=True, size=data_points.shape[0])
            means = np.mean(data_points[indices == 0], axis=0)
            for i in range(1, k):
                means = np.vstack((means, np.mean(data_points[indices == i], axis=0)))

            return means

        case "kmeans++":
            centers = data_points[np.random.choice(range(k))]

            dist_min = np.linalg.norm(data_points - centers, axis=1)
            pdf = dist_min / np.sum(dist_min)
            for i in range(k - 1):
                new_centroid = data_points[np.random.choice(data_points_len, replace=False, p=pdf), :]
                curr_dist = np.linalg.norm(data_points - new_centroid, axis=1)
                dist_min = np.minimum(dist_min, curr_dist)
                pdf = dist_min / np.sum(dist_min)
                centers = np.vstack((centers, new_centroid))

            return centers


def find_closest_centroids(data_points, centers):
    data_points_len = data_points.shape[0]
    k = centers.shape[0]
    close = np.zeros(data_points_len, dtype=int)

    for i in range(data_points_len):
        min_dist = float('inf')
        for j in range(k):
            curr_dist = np.dot(data_points[i] - centers[j], data_points[i] - centers[j])
            if min_dist > curr_dist:
                close[i] = j
                min_dist = curr_dist

    return close


def compute_new_centroids(data_points, close, k):
    data_type_len = data_points.shape[1]
    centers = np.zeros((k, data_type_len))

    for i in range(k):
        points = data_points[close == i]
        centers[i] = np.mean(points, axis=0)

    return centers


def kmeans_clustering(data_points, k, initialize_method, iteration=1):
    centroids = centroids_initialization(data_points, k, initialize_method)
    closest = np.zeros(data_points.shape[0])
    for i in range(iteration):
        print("K-Means iteration %d/%d" % (i+1, iteration))
        closest = find_closest_centroids(data_points, centroids)
        centroids = compute_new_centroids(data_points, closest, k)
    return centroids, closest


original_img = plt.imread("images/nw4.png")
original_img = original_img / 255
X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], original_img.shape[2]))

"""
K = 2
max_iteration = 4
initialization_method = "kmeans++"
centroid, idx = kmeans_clustering(X_img, K, initialization_method, max_iteration)

X_compressed = centroid[idx, :]
X_compressed = np.reshape(X_compressed, original_img.shape)

plt.axis('off')
plt.imshow (X_compressed * 255)

# plt.show()
plt.savefig('nw4', bbox_inches='tight')
"""
