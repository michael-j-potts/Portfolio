class KNN:
    def __init__(self):
        self.x = None
        self.y = None

    def train(self, x, y):
        self.x = x
        self.y = y
    def predict(self, x, k):
        dist = [(self.distance(x, TrainPoint), label) for TrainPoint, label in zip(self.x, self.y)]
        neighbours = sorted(dist)[:k]
        return sum(label for _, label in neighbours) / k

    def GetDistance(point1, point2):
        #Here we calculate the euclidean distance between points
        return((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5