import matplotlib.pyplot as plt

from kmeansClustering import *

img_location = "images/scene.png"
original_img = plt.imread(img_location)
original_img = original_img / 255
original_img_reshaped = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], original_img.shape[2]))

# initialize parameters
K = 2
max_iteration = 2

# choose initialization options
initialization_methods = ["forgy", "random partition", "kmeans++"]

# prepare for plots
fig, ax = plt.subplots(1, len(initialization_methods) + 1, figsize=(16, 4))
fig.suptitle("Comparison of K-means Image Compressions Using Different Initializations \n with {} clusters and {} "
             "iterations".format(K, max_iteration))
plt.axis('off')

# record errors of image compression using Frobenius norm
Error = [float('inf')] * len(initialization_methods)

# show original image
ax[0].imshow(original_img*255)
ax[0].set_title('Original')
ax[0].set_axis_off()

# compress image with different initialization

for i, initialization in enumerate(initialization_methods, 1):
    X_img = original_img_reshaped
    centroid, idx = kmeans_clustering(X_img, K, initialization, max_iteration)
    compressed_img = centroid[idx, :]
    Error[i - 1] = np.amax(np.linalg.norm(compressed_img - original_img_reshaped, float('inf'))) * 255
    compressed_img = np.reshape(compressed_img, original_img.shape)
    ax[i].imshow(compressed_img * 255)
    ax[i].set_title('{}'.format(initialization))
    ax[i].set_axis_off()

# compare the compressed matrix with original matrix
for initialization, error in zip(initialization_methods, Error):
    print("The error using {} initialization is {}".format(initialization, error))

plt.show()

