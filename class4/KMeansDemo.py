import random
import math
import matplotlib.pyplot as plt
class myKmeans():
    # 初始化
    def __init__(self, knums, iter, inputs):
        self.knums = knums    #设置聚类中心个数
        self.iter = iter      #设置最大迭代次数
        self.inputs = inputs  #原始输入数据
        self.label = []       #初始化分类标签

    # 随机初始化聚类中心
    def _initialize_centroids(self):
        self.centroids = random.sample(self.inputs, self.knums)

    # 根据分类中心对数据进行分类
    def _assign(self):
        for i in self.inputs:
            distancefromcentroids = []
            for j in self.centroids:
                distance = math.sqrt(math.pow(i[0] - j[0], 2) + math.pow(i[1] - j[1], 2))
                distancefromcentroids.append(distance)
            self.label.append(distancefromcentroids.index(min(distancefromcentroids)))

    # 更新聚类中心
    def _update_centroids(self):
        centroids_new = [[0, 0] for i in range(self.knums)]
        cluster_nums = [0 for i in range(self.knums)]
        for i in range(len(self.inputs)):
            centroids_new[self.label[i]][0] += self.inputs[i][0]
            centroids_new[self.label[i]][1] += self.inputs[i][1]
            cluster_nums[self.label[i]] += 1
        for i in range(self.knums):
            centroids_new[i][0] /= cluster_nums[i]
            centroids_new[i][1] /= cluster_nums[i]
        return centroids_new

    # 进行kmeans聚类
    def process(self):
        self._initialize_centroids()
        centroids_old = self.centroids
        for i in range(self.iter):
            self._assign()
            centroids_new = self._update_centroids()
            distance = 0
            for j in range(len(centroids_new)):
                distance += math.sqrt(math.pow(centroids_new[j][0] - centroids_old[j][0], 2) + math.pow(centroids_new[j][1] - centroids_old[j][1], 2))
            if distance < 0.1:
                break
            centroids_old = centroids_new
        return centroids_old


def genData(knums, samples, radius):
    centroids_x = [random.randint(-10, 10) for i in range(knums)]
    centroids_y = [random.randint(-10, 10) for i in range(knums)]
    x = []
    y = []
    for i in range(samples):
        index = random.choice(range(knums))
        x.append(centroids_x[index] + (random.random() - 0.5) * radius)
        y.append(centroids_y[index] + (random.random() - 0.5) * radius)
    return x,y

if __name__ == "__main__":
    x,y = genData(3, 30, 5)
    raw_data = list(zip(x, y))    
    myTest = myKmeans(3, 100, raw_data)
    centroids = myTest.process()
    #stymap = {0: '.', 1: '*', 2: '-'}
    colmap = {0: 'r', 1: 'g', 2: 'b'}
    for i in range(len(raw_data)):
        plt.plot(raw_data[i][0], raw_data[i][1], '*', color = colmap[myTest.label[i]])
    for i in centroids:
        plt.plot(i[0], i[1], 'o', color = 'black')
    plt.show()
