import numpy as np
import random
import pandas as pd

class GA:
    def __init__(self, distance_matrix, pop_size = 100, elite_size = 20, mutation_rate =0.01, generations= 500):
        self.distance_matrix = distance_matrix
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.cities = list(range(len(distance_matrix)))

    def create_route(self):
        return random.sample(self.cities, len(self.cities))

    def initial_population(self):
        return [self.create_route() for _ in range(self.pop_size)]

    def compute_fitness(self, route):
        return 1 / np.sum([self.distance_matrix[route[i-1]][route[i]] for i in range(len(route))])

    def rank_routes(self, population):
        fitness_results = [(i, self.compute_fitness(population[i])) for i in range(len(population))]
        return sorted(fitness_results, key = lambda x: x[1], reverse = True)

    def selection(self, pop_ranked, population):
        selection_results = pop_ranked[:self.elite_size]
        df = pd.DataFrame(np.array(pop_ranked), columns=["Index","Fitness"])
        df['cum_sum'] = df.Fitness.cumsum()
        df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
        
        for i in range(len(pop_ranked) - self.elite_size):
            pick = 100*random.random()
            for i in range(len(pop_ranked)):
                if pick <= df.iat[i,3]:
                    selection_results.append(pop_ranked[i])
                    break
        return [population[i] for i, _ in selection_results]

    def breed(self, parent1, parent2):
        child = []
        childP1 = []
        childP2 = []
        
        geneA = int(random.random() * len(parent1))
        geneB = int(random.random() * len(parent1))
        
        startGene = min(geneA, geneB)
        endGene = max(geneA, geneB)

        for i in range(startGene, endGene):
            childP1.append(parent1[i])
            
        childP2 = [item for item in parent2 if item not in childP1]

        child = childP1 + childP2
        return child

    def breed_population(self, matingpool):
        children = []
        length = len(matingpool) - self.elite_size
        pool = random.sample(matingpool, len(matingpool))

        for i in range(self.elite_size):
            children.append(matingpool[i])
        
        for i in range(length):
            child = self.breed(pool[i], pool[len(matingpool)-i-1])
            children.append(child)
        return children

    def mutate(self, individual):
        for swapped in range(len(individual)):
            if(random.random() < self.mutation_rate):
                swap_with = int(random.random() * len(individual))
                
                city1 = individual[swapped]
                city2 = individual[swap_with]
                
                individual[swapped] = city2
                individual[swap_with] = city1
        return individual

    def mutate_population(self, population):
        mutated_pop = []
        
        for ind in range(len(population)):
            mutated_ind = self.mutate(population[ind])
            mutated_pop.append(mutated_ind)
        return mutated_pop

    def two_opt(self, route):
        best = route
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j - i == 1: continue  # changes nothing, skip then
                    new_route = route[:]
                    new_route[i:j] = route[j - 1:i - 1:-1]  # this is the 2-optSwap
                    if self.compute_fitness(new_route) > self.compute_fitness(best):
                        best = new_route
                        improved = True
            route = best
        return best

    def next_generation(self, current_gen):
        pop_ranked = self.rank_routes(current_gen)
        selection_results = self.selection(pop_ranked, current_gen)
        matingpool = selection_results
        children = self.breed_population(matingpool)
        next_gen = self.mutate_population(children)
        next_gen = [self.two_opt(ind) for ind in next_gen]  # apply 2-opt local search
        return next_gen

    def run(self):
        pop = self.initial_population()
        print("Initial distance: " + str(1 / self.rank_routes(pop)[0][1]))
        
        last_five_generations_fitness = [0, 0, 0, 0, 0]
        
        for i in range(self.generations):
            pop = self.next_generation(pop)
            current_fitness = self.rank_routes(pop)[0][1]
            last_five_generations_fitness[i % 5] = current_fitness
            
            if i >= 4 and len(set(last_five_generations_fitness)) == 1:
                print("Fitness didn't change for 5 generations. Stopping...")
                break
        
        print("Final distance: " + str(1 / self.rank_routes(pop)[0][1]))
        best_route_index = self.rank_routes(pop)[0][0]
        best_route = pop[best_route_index]
        best_distance = 1 / self.rank_routes(pop)[0][1]
        return best_distance, best_route