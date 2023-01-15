import random

def KMeans(data, k):
    centroids = Initialize(data, k)
    while True:
        prev = centroids
        y = GetLabels(data, centroids)
        centroids = Update(data, y, k)
        if Settled(prev, centroids):
            break
    return y

def Initialize(data, k):
    XMin = YMin = float('inf')
    XMax = YMax = float('-inf')
    for point in data:
        XMin = min(point[0], XMin)
        YMin = min(point[1], YMin)
        XMax = max(point[0], XMax)
        YMax = max(point[1], YMax)
    centroids = []
    for i in range(k):
        centroids.append([RandomSample(XMin, XMax), RandomSample(YMin, YMax)])
    return centroids

def RandomSample(min, max):
    return min + (max-min) * random.random()

def GetLabels(data, centroids):
    y = []
    for point in data:
        min = float('inf')
        label = None
        for i, centroid in enumerate(centroids):
            dist = GetDistance(point, centroid)
            if min > dist:
                min = dist
                label = i
        y.append(label)
    return y

def GetDistance(point1, point2):
    #Here we calculate the euclidean distance between points
    return((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def Update(points, labels, k):
    updated = [[0,0] for i in range(k)]
    counts = [0] * k
    for point, label in zip(points, labels):
        updated[label][0] += point[0]
        updated[label][1] += point[1]
        counts[label] += 1
    for i, (x,y) in enumerate(updated):
        updated[i] = (x / counts[i], y / counts[i])
    return updated

def Settled(prev, updated, threshold = 1e-4):
    adjustment = 0
    for PrevPoint, UpdatedPoint in zip(prev, updated):
        adjustment += GetDistance(PrevPoint, UpdatedPoint)
    return adjustment < threshold