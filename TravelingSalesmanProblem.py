import csv, pickle, os, codecs

import numpy as np
from urllib.request import urlopen

import matplotlib.pyplot as plt

class TSP:
    """
    This class encapsulates the traveling salesman problem.
    City coordinates read from online website and data is serialized to disk.

    Augments:
        name: The name of the corresponding TSPLIB problem = 'burma14' or 'bayg29'
    """
    def __init__(self, name):
        self.name = name
        self.locations = []
        self.distances = []
        self.tspSize = 0

        # initializing the data
        self.__initData()

    def __len__(self):
        return self.tspSize

    def __initData(self):
        try:
            self.locations = pickle.load(open(os.path.join("tsp-data", self.name + "-loc.pickle"), "rb"))
            self.distances = pickle.load(open(os.path.join("tsp-data", self.name + "-dist.pickle"), "rb"))
            print("successfully load the coordinates!!!")
        except(OSError, IOError):
            pass

        if not np.array(self.locations).any() or not np.array(self.distances).any():
            self.__createData()

        self.tspSize = len(self.locations)

    def __createData(self):
        self.locations = []

        with urlopen("http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/" + self.name + ".tsp") as file:
            reader = csv.reader(codecs.iterdecode(file, 'utf-8'), delimiter=" ", skipinitialspace=True)

            # skip lines until one of these lines is found
            for row in reader:
                if row[0] in ('DISPLAY_DATA_SECTION', 'NODE_COORD_SECTION'):
                    break

            # read data line until 'EOF' found

            for row in reader:
                if row[0] != 'EOF':
                    # remove index from beginning of the line
                    del row[0]

                    self.locations.append(np.asarray(row, dtype=np.float32))
                else:
                    break

            self.tspSize = len(self.locations)

            print(f"length = {self.tspSize}, location = {self.locations}")

            # initializing distance matrix by filling it with 0's:
            self.distances = np.zeros((self.tspSize, self.tspSize), dtype=np.float32)
            print(f"Shape of distances: {self.distances.shape}")

            for i in range(self.tspSize):
                for j in range(i+1, self.tspSize):
                    distance = np.linalg.norm(self.locations[j]-self.locations[i])
                    self.distances[i][j] = distance
                    self.distances[j][i] = distance
                    print(f"{i}, {j}: location1 = {self.locations[i]}, location2 = {self.locations[j]} => distance = {distance}")

            if not os.path.exists("tsp-data"):
                os.mkdir("tsp-data")

            pickle.dump(self.locations, open(os.path.join("tsp-data", self.name + "-loc.pickle"), "wb"))
            pickle.dump(self.distances, open(os.path.join("tsp-data", self.name + "-dist.pickle"), "wb"))



    def getTotalDistance(self, indices):
        distance = self.distances[indices[-1]][indices[0]]

        for i in range(len(indices)-1):
            distance += self.distances[indices[i]][indices[i+1]]

        return distance

    def plotData(self, indices):
        plt.scatter(*zip(*self.locations), marker='o', color='red')

        locs = [self.locations[i] for i in indices]
        locs.append(locs[0])

        plt.plot(*zip(*locs), linestyle='-', color='blue')

        return plt


def main():
    tsp = TSP("bayg29")
    optimalSolution = [0, 27, 5, 11, 8, 25, 2, 28, 4, 20, 1, 19, 9, 3, 14, 17, 13, 16, 21, 10, 18, 24, 6, 22, 7, 26, 15, 12, 23]

    print(f"Problem name: {tsp.name}")
    print(f"Optimal solution: {optimalSolution}")
    print(f"Optimal distance: {tsp.getTotalDistance(optimalSolution)}")

    plot = tsp.plotData(optimalSolution)
    plot.show()


if __name__ == '__main__':
    main()