import random
import pandas as pd
import numpy as np
import math
from numpy.random.mtrand import randint


tsp = [[0, 400, 500, 300], [400, 0, 300, 500],
       [500, 300, 0, 400], [300, 500, 400, 0]]


class SimmulatedAnealing:
    def __init__(self):
        self.data = SimmulatedAnealing.load_data()
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

    def getNeighboursInsert():
        pass

    def getNeighboursSwap(self, solution):
        neighbours = []
        for i in range(len(solution)):
            for j in range(i+1, len(solution)):
                neighbour = solution.copy()
                neighbour[i] = solution[j]
                neighbour[j] = solution[i]
                neighbours.append(neighbour)
        # print("random neighbour", random.choice(neighbours))
        # zwraca liste sąsiadów
        return neighbours

    def getBestNeighbour(self, neighbours):
        # bestRouteLength = self.routeLength(neighbours[0])
        # bestNeighbour = neighbours[0]
        # for neighbour in neighbours:
        #     currentRouteLength = self.routeLength(neighbour)
        #     if(currentRouteLength < bestRouteLength):
        #         bestRouteLength = currentRouteLength
        #         bestNeighbour = neighbour
        # return bestNeighbour, bestRouteLength
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

    def SimmulatedAnealing(self, maxIteration, temp):
        currentSolution = self.randomSolution()
        print(currentSolution)
        currentRouteLength = self.routeLength(currentSolution)
        print(currentRouteLength)
        neighbours = self.getNeighboursSwap(currentSolution)
        bestNeighbour, bestNeighbourRouteLength = self.getBestNeighbour(
            neighbours)
        iterator = 0
        while True:
            if(iterator < maxIteration):
                neighbours = self.getNeighboursSwap(currentSolution)
                bestNeighbour, bestNeighbourRouteLength = self.getBestNeighbour(
                    neighbours)
                if(bestNeighbourRouteLength < currentRouteLength):
                    currentSolution = bestNeighbour
                    currentRouteLength = bestNeighbourRouteLength

                    print(currentSolution, currentRouteLength)
                    print(bestNeighbour, bestNeighbourRouteLength)
                else:  # nowa >= aktualnego rozwiązania
                    dE = bestNeighbourRouteLength - currentRouteLength
                    p = math.exp(-dE/temp)
                    rand = random.random()
                    if(rand < p):
                        currentSolution = bestNeighbour
                        currentRouteLength = bestNeighbourRouteLength

                        print(currentSolution, currentRouteLength)
                        print(bestNeighbour, bestNeighbourRouteLength)

                iterator = iterator + 1
                wspolczynnik = 0.99
                T = wspolczynnik*temp  # redukcja geometryczna
                temp = T
                # wspolczynnik = 0.99
                # T = temp/(1+wspolczynnik*temp) # powolny spadek
                # temp = T
                print("iteration", iterator)
            else:
                break
        # print(neighbours[0], neighbours[1], neighbours[2])
        return currentSolution, currentRouteLength

    def SimmulatedAnealingInsert(self, maxIteration, temp):
        currentSolution = self.randomSolution()
        print(currentSolution)
        currentRouteLength = self.routeLength(currentSolution)
        print(currentRouteLength)
        bestNeighbour, bestNeighbourRouteLength = self.InsertN(currentSolution)
        iterator = 0
        while True:
            if(iterator < maxIteration):
                bestNeighbour, bestNeighbourRouteLength = self.InsertN(
                    currentSolution)
                if(bestNeighbourRouteLength < currentRouteLength):
                    currentSolution = bestNeighbour
                    currentRouteLength = bestNeighbourRouteLength

                    print(currentSolution, currentRouteLength)
                    print(bestNeighbour, bestNeighbourRouteLength)
                else:  # nowa >= aktualnego rozwiązania
                    dE = bestNeighbourRouteLength - currentRouteLength
                    p = math.exp(-dE/temp)
                    rand = random.random()
                    if(rand < p):
                        currentSolution = bestNeighbour
                        currentRouteLength = bestNeighbourRouteLength

                        print(currentSolution, currentRouteLength)
                        print(bestNeighbour, bestNeighbourRouteLength)

                iterator = iterator + 1
                wspolczynnik = 0.99
                T = wspolczynnik*temp  # redukcja geometryczna
                temp = T
                # wspolczynnik = 0.99
                # T = temp/(1+wspolczynnik*temp) # powolny spadek
                # temp = T
                print("iteration", iterator)
            else:
                break
        # print(neighbours[0], neighbours[1], neighbours[2])
        return currentSolution, currentRouteLength


TSP = SimmulatedAnealing()
bestSol, bestSolRoute = TSP.SimmulatedAnealingInsert(10000, 50)
print(bestSol, bestSolRoute)
