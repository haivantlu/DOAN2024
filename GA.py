import numpy as np
import random

class GA:
    def __init__(self, distance_matrix, population_size=100, mutation_rate=0.01, generations=50):
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.num_cities = len(distance_matrix)

    def create_individual(self):
        return [0] + random.sample(range(1, self.num_cities), self.num_cities - 1)

    def create_population(self):
        return [self.create_individual() for _ in range(self.population_size)]

    def fitness(self, individual):
        return sum([self.distance_matrix[individual[i-1]][individual[i]] for i in range(self.num_cities)])

    def rank_population(self, population):
        return sorted([(self.fitness(i), i) for i in population], key=lambda x: x[0], reverse=False)

    def selection(self, ranked_population):
        return ranked_population[0][1], ranked_population[1][1]

    def crossover(self, parent1, parent2):
        child = [None]*self.num_cities
        start, end = sorted(random.sample(range(1, self.num_cities), 2))
        child[start:end] = [city for city in parent1 if city in parent2[start:end]]
        child = [city if city is not None else parent2[i] for i, city in enumerate(child)]
        return child

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            swap1, swap2 = random.sample(range(self.num_cities), 2)
            individual[swap1], individual[swap2] = individual[swap2], individual[swap1]
        return individual

    def evolve_population(self, population):
        ranked_population = self.rank_population(population)
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = self.selection(ranked_population)
            child = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child))
        return new_population

    def run(self):
        population = self.create_population()
        for _ in range(self.generations):
            population = self.evolve_population(population)
        best_individual = self.rank_population(population)[0]
        return best_individual