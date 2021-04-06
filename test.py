import pandas as pd
import numpy as np
import random

# визуализация
import pylab as plt

# Train data generator
def generateData(numberOfClassEl, numberOfClasses):
    data = []
    for classNum in range(numberOfClasses):
        # Choose random center of 2-dimensional gaussian
        centerX, centerY = random.random() * 5.0, random.random() * 5.0
        # Choose numberOfClassEl random nodes with RMS=0.5
        for rowNum in range(numberOfClassEl):
            data.append([[random.gauss(centerX, 0.5), random.gauss(centerY, 0.5)], classNum])
    return data


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


n = 10
k = 6
data = []
data = generateData(n, k)
x, y = [], []
color_map = {
    0: 'pink',
    1: 'b',
    2: 'g',
    3: 'y',
    4: 'm',
    5: 'c'
}
data = []
data = generateData(n, k)
x, y = [], []
clusters = {}
for i in range(len(data)):
    cls = data[i][1]

    if (not cls in clusters.keys()):
        clusters[cls] = []
    clusters[cls].append((data[i][0][0], data[i][0][1]))

for num, cluster in clusters.items():
    x, y = [], []
    for cluster_xy in cluster:
        x.append(cluster_xy[0])
        y.append(cluster_xy[1])
    plt.scatter(x, y, color=color_map[num])

plt.show()
min_x, max_x = np.min(x), np.max(x)
min_y, max_y = np.min(y), np.max(y)

# # создадим 5 новых точек
# for i in range(5):
#     x_new = min_x + np.random.random() * (max_x - min_x)
#     y_new = min_y + np.random.random() * (max_y - min_y)
#
#     plt.scatter(x_new, y_new, color='r')
#
#     distance = []
#     for j in range(n * k):
#         distance.append([data[j][1], dist(data[j][0][0], data[j][0][1], x_new, y_new)])
#
#         # вычисляем k точек, расстояние которых мин до новой
#     points_with_min_distances = []
#     clusters = []
#     min = 1000
#     min_cl = 12
#     for j in range(k):
#         for i in range(len(distance)):
#             if (len(points_with_min_distances) == 0):
#                 if (distance[i][1] < min):
#                     min = distance[i][1]
#                     min_cl = distance[i][0]
#             else:
#                 if ((distance[i][1] < min) & (distance[i][1] > points_with_min_distances[j - 1])):
#                     min = distance[i][1]
#                     min_cl = distance[i][0]
#         points_with_min_distances.append(min)
#         clusters.append(min_cl)
#         min = 1000
#     print("\n\nТочка: (", x_new, ", ", y_new, ")")
#     print("\n10 минимальных расстояний от точки: ")
#     print(points_with_min_distances)
#
#     import numpy as np
#
#     bc = np.bincount(clusters)
#
#     print("Кластеры точек, у которых минимальное расстояние до выбранной: ", clusters, "\nКластер выбранной точки:",
#           bc.argmax())

