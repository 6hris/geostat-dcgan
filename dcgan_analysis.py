from PIL import Image
import os
import numpy as np

def calculations(image_folder):
    # calculate mean and variance
    means = []
    variances = []

    for name in os.listdir(image_folder):
        path = os.path.join(image_folder, name)
        image = Image.open(path).convert('L')
        image_normalized = np.array(image, dtype=np.float32) / 255.0

        mean = np.mean(image_normalized)
        variance = np.var(image_normalized)

        means.append(mean)
        variances.append(variance)

    return np.mean(means), np.mean(variances)

image_folder = 'images'
avg_mean, avg_variance = calculations(image_folder)
print(f"Average Mean: {avg_mean}, Average Variance: {avg_variance}")