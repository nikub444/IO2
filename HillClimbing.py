import random
from numpy.random.mtrand import randint
import pandas as pd
import numpy as np

tsp = [[0, 400, 500, 300], [400, 0, 300, 500],
       [500, 300, 0, 400], [300, 500, 400, 0]]


class HillClimbing:
    def __init__(self):
        self.data = HillClimbing.load_data()
        self.rows = self.data.shape[0]
        self.cols = self.data.shape[1]

    @staticmethod
    def load_data():
        df = pd.read_csv("TSP48.csv", sep=";", header=None)
        return df

    def randomSolution(self):
        solution = np.random.permutation(self.rows)
        return solution

    def routeLength(self, solution):
        routeLength = 0
        for i in range(self.rows):
            # print(type(self.data[solution[i-1]][solution[i]]))
            routeLength += self.data[solution[i-1]][solution[i]]
            # print(i, solution[i-1], solution[i])

        return routeLength

    def ListComparison(self, list1, list2):
        for el1, el2 in zip(list1, list2):
            if(el1 == el2):
                pass
            else:
                return False
        return True

    def getNeighboursInsert(self, solution):
        neighbours = []
        solution = list(solution)
        for i in range(len(solution)):
            for j in range(len(solution)):
                neighbour = solution.copy()

                item = neighbour.pop(i)
                neighbour.insert(j, item)
                duplicate = False
                for x in neighbours:
                    if(self.ListComparison(neighbour, x)):
                        duplicate = True
                if(duplicate):
                    continue
                else:
                    neighbours.append(neighbour)
        return neighbours

    def getNeighboursSwap(self, solution):
        neighbours = []
        for i in range(len(solution)):
            for j in range(i+1, len(solution)):
                neighbour = solution.copy()
                neighbour[i] = solution[j]
                neighbour[j] = solution[i]
                neighbours.append(neighbour)
        return neighbours

    def getBestNeighbour(self, neighbours):
        # bestRouteLength = self.routeLength(neighbours[0])
        # bestNeighbour = neighbours[0]
        # for neighbour in neighbours:
        #     currentRouteLength = self.routeLength(neighbour)
        #     if(currentRouteLength < bestRouteLength):
        #         bestRouteLength = currentRouteLength
        #         bestNeighbour = neighbour
        randNeighbour = random.choice(neighbours)
        bestRouteLength = self.routeLength(randNeighbour)
        bestNeighbour = randNeighbour
        return bestNeighbour, bestRouteLength

    def InsertN(self, solution):
        if(isinstance(solution, np.ndarray)):
            solution = solution.tolist()
        while True:
            id1 = randint(0, self.rows - 1)
            id2 = randint(0, self.rows - 1)

            if id1 != id2:
                break
        val = solution[id1]
        solution.insert(id2, val)
        if id1 < id2:
            solution.pop(id1)
        else:
            solution.pop(id1+1)

        bestRouteLength = self.routeLength(solution)
        bestNeighbour = solution
        return bestNeighbour, bestRouteLength

    def HillClimb(self, multistart, maxIteration):
        bestResult = []
        bestRoute = []
        for _ in range(multistart):
            currentSolution = self.randomSolution()
            print(currentSolution)
            currentRouteLength = self.routeLength(currentSolution)
            print(currentRouteLength)
            neighbours = self.getNeighboursSwap(currentSolution)
            bestNeighbour, bestNeighbourRouteLength = self.getBestNeighbour(
                neighbours)
            iterator = 0
            # while bestNeighbourRouteLength < currentRouteLength:
            for x in range(maxIteration):
                if(bestNeighbourRouteLength < currentRouteLength):
                    currentSolution = bestNeighbour
                    currentRouteLength = bestNeighbourRouteLength
                    neighbours = self.getNeighboursSwap(currentSolution)
                    bestNeighbour, bestNeighbourRouteLength = self.getBestNeighbour(
                        neighbours)
                    print(bestNeighbour, bestNeighbourRouteLength)
                    iterator = iterator + 1
                    print("iteration", iterator)
                else:
                    print("iteration", iterator)
                    neighbours = self.getNeighboursSwap(currentSolution)
                    bestNeighbour, bestNeighbourRouteLength = self.getBestNeighbour(
                        neighbours)
                    iterator = iterator + 1
                print("currentLen", currentRouteLength,
                      "new", bestNeighbourRouteLength)
            bestResult.append(currentSolution)
            bestRoute.append(currentRouteLength)
        # print(neighbours[0], neighbours[1], neighbours[2])
        return bestResult, bestRoute

    def HillClimbInsert(self, multistart, maxIteration):
        bestResult = []
        bestRoute = []
        for _ in range(multistart):
            currentSolution = self.randomSolution()
            print(currentSolution)
            currentRouteLength = self.routeLength(currentSolution)
            print(currentRouteLength)
            bestNeighbour, bestNeighbourRouteLength = self.InsertN(
                currentSolution)
            iterator = 0
            # while bestNeighbourRouteLength < currentRouteLength:
            for x in range(maxIteration):
                if(bestNeighbourRouteLength < currentRouteLength):
                    currentSolution = bestNeighbour
                    currentRouteLength = bestNeighbourRouteLength
                    bestNeighbour, bestNeighbourRouteLength = self.InsertN(
                        currentSolution)
                    print(bestNeighbour, bestNeighbourRouteLength)
                    iterator = iterator + 1
                    print("iteration", iterator)
                else:
                    print("iteration", iterator)
                    bestNeighbour, bestNeighbourRouteLength = self.InsertN(
                        currentSolution)
                    iterator = iterator + 1
                print("currentLen", currentRouteLength,
                      "new", bestNeighbourRouteLength)
            bestResult.append(currentSolution)
            bestRoute.append(currentRouteLength)
        # print(neighbours[0], neighbours[1], neighbours[2])
        return bestResult, bestRoute


TSP = HillClimbing()
bestResult, bestRoute = TSP.HillClimbInsert(1, 100000000)
print(bestResult, bestRoute)
# print(TSP.getNeighboursInsert([1, 2, 3]))
