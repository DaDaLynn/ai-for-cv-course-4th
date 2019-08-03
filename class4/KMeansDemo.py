import random
import matplotlib.pyplot as plt
class myKmeans():
    def __init__(self, knums, iter):
        pass

    def initialize_centroids(self):
        pass

    def assign(self):
        pass

    def update_centroids(self):
        pass

def genData(knums, samples, radius):
    centroids_x = [random.randint(5, 10) for i in range(knums)]
    centroids_y = [random.randint(5, 10) for i in range(knums)]
    x = []
    y = []
    for i in range(samples):
        index = random.choice(range(knums))
        x.append(centroids_x[index] + (random.random() - 0.5) * radius)
        y.append(centroids_y[index] + (random.random() - 0.5) * radius)
    return x,y

if __name__ == "__main__":
    x,y = genData(3, 30, 5)
    plt.plot(x,y, '.')
    plt.show()
