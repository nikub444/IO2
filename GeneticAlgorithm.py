import pandas as pd
import numpy as np
from random import randint, uniform


class TSPGA:
    def __init__(self):
        self.iterations = 200
        self.tournament_size = 20
        self.elitism_no = 30
        self.data = TSPGA.load_data()
        self.rows = self.data.shape[0]
        self.cols = self.data.shape[1]
        self.population_size = 200
        self.mutation_rate = 0.2
        self.population = [np.random.permutation(
            self.rows) for _ in range(self.population_size)]
        self.evaluation = [0 for _ in range(len(self.population))]

    def rank_selection(self):
        permutation = sorted(range(len(self.evaluation)),
                             key=lambda k: self.evaluation[k])
        self.population = [self.population[i] for i in permutation]
        cumulative_sum = [0 for _ in range(self.population_size)]
        cumulative_sum[0] = 1
        for i in range(1, self.population_size):
            cumulative_sum[i] = cumulative_sum[i - 1] + i + 1
        rand_val = randint(0, cumulative_sum[-1])
        index = 0
        while rand_val > cumulative_sum[index]:
            index = index + 1
        return self.population[index]

    def pick_parent_roulette(self):
        sum = np.sum(self.evaluation)
        start_id = self.evaluation[0] / sum
        stop_id = self.evaluation[1] / sum
        board = [0 for _ in range(len(self.population))]
        for x in range(len(self.population)):
            board[x] = np.around(start_id, decimals=7)
            start_id = start_id + stop_id
            stop_id = self.evaluation[x] / sum

        board[-1] = 1

        rand_val = uniform(0, 1)

        parent_id = 0
        while rand_val >= board[parent_id]:
            parent_id = parent_id + 1
        return self.population[parent_id]

    @staticmethod
    def load_data():
        df = pd.read_csv("TSP48.csv", sep=";", header=None)
        return df

    def evolve(self):
        for iteration in range(self.iterations):
            new_population = []
            self.fitness()
            if self.elitism_no > 0:
                new_population = new_population + self.pick_top_indiviuals()
            while len(new_population) < self.population_size:
                p1 = self.pick_parent_tournament()
                p2 = self.pick_parent_tournament()
                [ch1, ch2] = self.order_one_crossover(p1, p2)
                ch1 = self.mutation_index(ch1)
                ch2 = self.mutation_index(ch2)
                new_population = new_population + [ch1, ch2]

            self.population = new_population

    def pick_top_indiviuals(self):
        permutation = sorted(range(len(self.evaluation)),
                             key=lambda k: self.evaluation[k])
        self.population = [self.population[i] for i in permutation]
        self.evaluation = [self.evaluation[i] for i in permutation]

        return self.population[self.population_size-self.elitism_no-1: self.population_size-1]

    def print_best_fitness(self, evaluation):
        print("Average: ", np.sum(evaluation) / len(evaluation))
        print("Best: ", np.min(evaluation))
        print()

    def pick_parent_tournament(self):
        perm = np.random.permutation(self.population_size)[
            0:self.tournament_size]
        best = 0
        best_id = 0
        for id in perm:
            if self.evaluation[id] > best:
                best = self.evaluation[id]
                best_id = id
        return self.population[best_id]

    def mutation_swap(self, child):
        if uniform(0, 1) > self.mutation_rate:
            return child

        while True:
            id1 = randint(0, self.rows - 1)
            id2 = randint(0, self.rows - 1)
            if id1 != id2:
                break
        tmp = child[id1]
        child[id1] = child[id2]
        child[id2] = tmp

        return child

    def fitness(self):
        def min_max(array):
            mx = np.max(array)
            mn = np.min(array)
            scaled_array = (6 - ((array - mn) / (mx - mn) * 4 + 1)) ** 2
            return scaled_array

        for genome_id in range(len(self.population)):
            fitness = 0
            genome = self.population[genome_id]
            for i in range(self.rows):
                if i == self.rows - 1:
                    fitness = fitness + self.data.iloc[genome[i], genome[0]]
                else:
                    fitness = fitness + self.data.iloc[genome[i], genome[i+1]]
            self.evaluation[genome_id] = fitness

        self.print_best_fitness(self.evaluation)

        self.evaluation = min_max(self.evaluation)

    def mutation_index(self, child: np.ndarray):
        if uniform(0, 1) > self.mutation_rate:
            return child

        while True:
            id1 = randint(0, self.rows - 1)
            id2 = randint(0, self.rows - 1)

            if id1 != id2:
                break

        val = child[id1]
        child = np.insert(child, id2, val)

        if id1 < id2:
            child = np.delete(child, id1)
        else:
            child = np.delete(child, id1+1)
        return child

    def order_one_crossover(self, parent1, parent2):
        size = len(parent1)

        child1 = np.ones(shape=(size,), dtype=np.int) * (-1)
        child2 = np.ones(shape=(size,), dtype=np.int) * (-1)

        # picking random segment
        start_index = randint(0, size - 2)
        length = randint(1, size - start_index - 1)

        child1[start_index:start_index +
               length] = parent1[start_index:start_index + length]
        child2[start_index:start_index +
               length] = parent2[start_index:start_index + length]

        parent_idx = start_index + length
        child_idx = parent_idx
        child2_idx = parent_idx

        for i in range(size):
            if parent_idx + i == size:
                parent_idx = parent_idx - size
            elem = parent1[parent_idx + i]
            elem2 = parent2[parent_idx + i]

            if elem not in child2:
                child2[child_idx] = elem
                child_idx = child_idx + 1
                if child_idx == size:
                    child_idx = 0

            if elem2 not in child1:
                child1[child2_idx] = elem2
                child2_idx = child2_idx + 1

                if child2_idx == size:
                    child2_idx = 0

        return [child1, child2]

    @staticmethod
    def cycle_crossover(parent1, parent2):

        def check_for_repetition(array_of_cycles1, value):
            for cycle in array_of_cycles1:
                for item in cycle:
                    if value == item["value"]:
                        return True
            return False

        def find_idx(array, value):
            for i in range(len(array)):
                if array[i] == value:
                    return i

        array_of_cycles1 = []
        array_of_cycles2 = []

        for i in range(len(parent1)):
            if check_for_repetition(array_of_cycles1, parent1[i]):
                continue
            start_cycle_value = parent1[i]
            mapped_value = parent2[i]
            idx = find_idx(parent1, mapped_value)
            cycle1 = [{"value": start_cycle_value, "pos": i}]
            cycle2 = [{"value": mapped_value, "pos": i}]
            while mapped_value != start_cycle_value:
                first_parent_val = parent1[idx]
                cycle1.append({"value": first_parent_val, "pos": idx})
                mapped_value = parent2[idx]
                cycle2.append({"value": mapped_value, "pos": idx})
                idx = find_idx(parent1, mapped_value)

            array_of_cycles1.append(cycle1)
            array_of_cycles2.append(cycle2)

        child1 = [0 for _ in range(len(parent1))]
        child2 = [0 for _ in range(len(parent1))]

        for i in range(len(array_of_cycles1)):
            if uniform(0, 1) > 0.5:
                for item in array_of_cycles1[i]:
                    child1[item["pos"]] = item["value"]
                for item in array_of_cycles2[i]:
                    child2[item["pos"]] = item["value"]
            else:
                for item in array_of_cycles2[i]:
                    child1[item["pos"]] = item["value"]
                for item in array_of_cycles1[i]:
                    child2[item["pos"]] = item["value"]

        return [child1, child2]


tsp = TSPGA()
tsp.evolve()
