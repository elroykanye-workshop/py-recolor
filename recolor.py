import sys
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class Recolor:
    def __init__(self, image, k):
        self.image = image
        self.K = k

    def recolor_image(self):
        cond = int(sys.argv[3])
        width, height = self.image.size

        # If clusters' initial coordinate chosen by clicking on image.
        if cond == 1:
            plt.imshow(self.image)
            points = plt.ginput(self.K, show_clicks=True)

        # If clusters' initial coordinate chosen by randomly.
        else:
            points = []
            for i in range(0, self.K):
                points.append([])
                x = int(np.random.uniform(0, width))
                y = int(np.random.uniform(0, height))
                points[i].append(x)
                points[i].append(y)

        return self.k_means_algo(points, width, height)

    def k_means_algo(self, points, width, height):
        """Converts an input image into a matrix of the image's pixels"""
        assigned_clusters = []
        pixels = np.array(self.image.getdata())

        # Picks K initial colors.
        cluster_values = np.zeros(shape=(self.K, 3))

        for i in range(0, self.K):
            cluster_values[i] = self.image.getpixel((int(points[i][0]), int(points[i][1])))

        for counter in range(0, 10):
            temp = []
            # Finds pixels distances to each cluster.
            for i in range(0, self.K):
                temp.append(np.sum(np.power(pixels - cluster_values[i], 2), axis=1))

            distances = np.stack(temp, axis=1)
            # Assigns each pixel to the closest cluster.
            assigned_clusters = np.argmin(distances, axis=1)

            # Find clusters' new R, G, B values.
            for i in range(0, self.K):
                cluster_values[i] = np.mean(pixels[assigned_clusters == i], axis=0)

        return Image.fromarray(cluster_values[assigned_clusters].reshape(height, width, -1).astype('uint8'), 'RGB')


##
# Main function, reads image and K value from arguments.
# Calls necessary function to execute K-Means Algorithm.
# Then according to result it saves output image.
##
if __name__ == '__main__':
    im = Image.open(sys.argv[1])
    K = int(sys.argv[2])

    r = Recolor(im, K)
    new_image = r.recolor_image()
    new_image.save("output.png")

# You can test with images in the source root
