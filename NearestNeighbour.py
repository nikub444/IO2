import random
import pandas as pd
import numpy as np


class NearestNeighbour:
    def __init__(self):
        self.data = NearestNeighbour.load_data()
        self.rows = self.data.shape[0]
        self.cols = self.data.shape[1]
        self.cities = list(range(0, self.rows))

    @staticmethod
    def load_data():
        df = pd.read_csv("TSP48.csv", sep=";", header=None)
        return df

    def randomSolution(self):
        solution = []
        rand = np.random.randint(self.rows)
        solution.append(rand)  # od 0 do 47
        self.cities.remove(rand)
        return solution

    def addNode(self, solution):
        solution.append(self.cities[0])
        best = self.cities[0]
        bestrouteLength = self.routeLength(solution)
        solution.remove(self.cities[0])
        for i in self.cities:
            tempSolution = solution.copy()
            tempSolution.append(i)

            if(bestrouteLength > self.routeLength(tempSolution)):
                bestrouteLength = self.routeLength(tempSolution)
                best = i
            print("city", i, "tempsolution", tempSolution,
                  "best route", best, bestrouteLength, self.routeLength(tempSolution))
        return best

    def routeLength(self, solution):
        # print("solutions", solution[-2], solution[-1])
        routeLength = self.data[solution[-2]][solution[-1]]
        # print(i, solution[i-1], solution[i])
        return routeLength

    def totalRouteLength(self, solution):
        routeLength = 0
        for i in range(self.rows):
            # print(type(self.data[solution[i-1]][solution[i]]))
            routeLength += self.data[solution[i-1]][solution[i]]
            # print(i, solution[i-1], solution[i])

        return routeLength

    def NearestNeighbour(self):
        currentSolution = self.randomSolution()
        print("SOl", currentSolution)
        # currentRouteLength = self.routeLength(currentSolution)
        # print("RL", currentRouteLength)
        while len(self.cities) > 0:
            print(len(self.cities), "self cities", self.cities)
            best = self.addNode(currentSolution)
            currentSolution.append(best)
            self.cities.remove(best)
        print("total solution", currentSolution, len(currentSolution))
        total = self.totalRouteLength(currentSolution)
        print("total solution", currentSolution, len(currentSolution), total)
        return currentSolution, total


TSP = NearestNeighbour()
solution, total = TSP.NearestNeighbour()
