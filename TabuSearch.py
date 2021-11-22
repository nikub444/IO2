import random
import pandas as pd
import numpy as np

tsp = [[0, 400, 500, 300], [400, 0, 300, 500],
       [500, 300, 0, 400], [300, 500, 400, 0]]


class TabuSearch:
    def __init__(self):
        self.data = TabuSearch.load_data()
        self.rows = self.data.shape[0]
        self.cols = self.data.shape[1]
        self.tabuList = []
        self.tabuWaitList = []
        self.tabuWait = 10
        self.maxTabuList = 11

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
        return neighbours

    def getBestNeighbour(self, neighbours):
        bestRouteLength = self.routeLength(neighbours[0])
        bestNeighbour = neighbours[0]
        for neighbour in neighbours:
            currentRouteLength = self.routeLength(neighbour)
            if(currentRouteLength < bestRouteLength):
                bestRouteLength = currentRouteLength
                bestNeighbour = neighbour
        return bestNeighbour, bestRouteLength

    def ListComparison(self, list1, list2):
        for el1, el2 in zip(list1, list2):
            if(el1 == el2):
                pass
            else:
                return False
        return True

        # for element in list1:
        #     counter = 0
        #     print(element, list2[counter])

        #     if(element == list2[counter]):
        #         counter = counter + 1
        #     else:
        #         return False
        # return True

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
                    if(self.ListComparison(neighbour, x) or neighbour[0] == x[-1] or x[0] == neighbour[-1]):
                        duplicate = True
                if(duplicate):
                    continue
                else:
                    neighbours.append(neighbour)
        return neighbours

    def NeighbourRemove(self, neighbours):
        neighboursList = []  # odejmuje wszystkich sąsiadów ztabu listy
        var = False
        for el in neighbours:
            for el2 in self.tabuList:
                if(self.ListComparison(el, el2)):
                    var = True
            if(var == False):
                neighboursList.append(el)
            var = False
        return neighboursList
        # neighboursList = []  # odejmuje jednego sąsiada
        # for el in neighbours:
        #     if(self.ListComparison(el, bestNeighbour)):
        #         continue
        #     neighboursList.append(el)
        # return neighboursList

    def Tabu(self, solution):
        inTabu = False
        print("Tabu", self.tabuList, self.tabuWaitList)

        for tabuMove in self.tabuList:
            # print(solution, tabuMove)
            if(self.ListComparison(tabuMove, solution)):
                inTabu = True
                print("solution", solution)
                print("TRUE")

        if(inTabu == False and len(self.tabuList) < self.maxTabuList):
            self.tabuList.append(solution)
            self.tabuWaitList.append(self.tabuWait)
        if(len(self.tabuWaitList)):
            self.tabuWaitList = [x - 1 for x in self.tabuWaitList]
        if(self.tabuWaitList[0] == 0 and len(self.tabuWaitList)):
            self.tabuWaitList.pop(0)
            self.tabuList.pop(0)
        return inTabu

    def TabuSearch(self,  maxIteration):

        currentSolution = self.randomSolution()
        print(currentSolution)
        currentRouteLength = self.routeLength(currentSolution)
        minSolution, minRoute = currentSolution, currentRouteLength
        print(currentRouteLength)
        neighbours = self.getNeighboursInsert(currentSolution)
        bestNeighbour, bestNeighbourRouteLength = self.getBestNeighbour(
            neighbours)
        for iteration in range(maxIteration):
            # if(bestNeighbourRouteLength < currentRouteLength):
            print("iteration", iteration)
            if(self.Tabu(bestNeighbour) == False):
                currentSolution = bestNeighbour
                print(currentSolution)
                currentRouteLength = bestNeighbourRouteLength
                print(currentRouteLength)
                neighbours = self.getNeighboursInsert(currentSolution)
                bestNeighbour, bestNeighbourRouteLength = self.getBestNeighbour(
                    neighbours)
            else:
                print("tutaj", bestNeighbour, bestNeighbourRouteLength)
                print(currentSolution, currentRouteLength)
                neighbours = self.getNeighboursInsert(currentSolution)
                # neighbours.remove(bestNeighbour)
                modifiedNeighbours = self.NeighbourRemove(
                    neighbours)

                bestNeighbour, bestNeighbourRouteLength = self.getBestNeighbour(
                    modifiedNeighbours)
                print("moded", bestNeighbour, bestNeighbourRouteLength)
            if(minRoute > currentRouteLength):
                minSolution, minRoute = currentSolution, currentRouteLength
        return minSolution, minRoute
        # else:
        #     print("iteration", iteration)
        #     if(self.Tabu(bestNeighbour) == False):
        #         currentSolution = bestNeighbour
        #         print(currentSolution)
        #         currentRouteLength = bestNeighbourRouteLength
        #         print(currentRouteLength)
        #         neighbours = self.getNeighboursSwap(currentSolution)
        #         bestNeighbour, bestNeighbourRouteLength = self.getBestNeighbour(
        #             neighbours)
        #     else:
        #         neighbours = self.getNeighboursSwap(currentSolution)
        #         # neighbours.remove(bestNeighbour)
        #         neighbours = self.NeighbourRemove(
        #             neighbours, bestNeighbour)
        #         bestNeighbour, bestNeighbourRouteLength = self.getBestNeighbour(
        #             neighbours)

        # inTabu = False
        # currentSolution = self.randomSolution()
        # print(currentSolution)
        # currentRouteLength = self.routeLength(currentSolution)
        # print(currentRouteLength)
        # neighbours = self.getNeighboursSwap(currentSolution)
        # bestNeighbour, bestNeighbourRouteLength = self.getBestNeighbour(
        #     neighbours)
        # iterator = 0
        # while bestNeighbourRouteLength < currentRouteLength:
        #     if(iterator < maxIteration):
        #         currentSolution = bestNeighbour
        #         currentRouteLength = bestNeighbourRouteLength
        #         neighbours = self.getNeighboursSwap(currentSolution)
        #         bestNeighbour, bestNeighbourRouteLength = self.getBestNeighbour(
        #             neighbours)
        #         print(bestNeighbour, bestNeighbourRouteLength)
        #         inTabu = self.Tabu(bestNeighbour)
        #         if(inTabu == True):
        #             while
        #             pass
        #         iterator = iterator + 1
        #         print("iteration", iterator)
        #     else:
        #         break

        # print(currentSolution, currentRouteLength)


TSP = TabuSearch()
# # bestResult, bestRoute = TSP.TabuSearch(1, 200)
# # print(bestResult, bestRoute)
minSolution, minRoute = TSP.TabuSearch(500)
print(minSolution, minRoute)


# def ListComparison(list1, list2):
#     for el1, el2 in zip(list1, list2):
#         if(el1 == el2):
#             pass
#         else:
#             return False
#     return True


# neighbours = [[5, 2, 3], [3, 2, 1], [2, 1, 3]]
# neighbours2 = [[1, 2, 3], [3, 2, 1], [2, 1, 3]]
# listn = []
# var = False
# for el in neighbours:
#     for el2 in neighbours2:
#         print(el, el2)
#         if(ListComparison(el, el2)):
#             print("True comparison", el, el2)
#             var = True
#     if(var == False):
#         listn.append(el)
# print(listn)
