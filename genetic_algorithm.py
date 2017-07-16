#! /usr/bin/env python
"""
A simple application of a genetic algorithm to a well known classification
problem -- the student exam problem

author: edelsonc
date: 7/14/2017
"""
import math
import random

class Population(object):
    """
    Generic population class for genetic algorithms
    """
    def __init__(self, n, loss_func, indv_gen, mut_func):
        """
        Arguments
        ---------
        n -- number of individuals in the population
        func -- loss function
        indv_gen -- function to generate individuals
        mut_func -- how parameters can be mutated
        """
        self.individuals = []
        self.fitness = None
        self.loss_func = loss_func
        self.indv_gen = indv_gen
        self.mut_func = mut_func
        self.n = n

    def generate_population(self,*args):
        """
        Function for generating the initial population using indv_gen

        *args are whatever arguments are needed for the indv_gen function
        """
        for i in range(self.n):
            self.individuals.append(self.indv_gen(*args))

    def eval_pop_fitness(self, *args):
        """
        Evaluates the fitness of each member of the population
        Creates a tuple of (fitness_score, individual_idx)
        """
        if self.fitness == None:
            self.fitness = [
                (self.loss_func(*arg), i) for i, arg in enumerate(zip(self.individuals, *args))
            ]
        else:
            for i,arg in enumerate(zip(self.individuals,*args)):
                self.fitness[i] = (self.loss_func(*arg), i)

    def average_fitness(self):
        """Averages the fitness scores across individuals"""
        total = 0
        count = 0
        for f in self.fitness:
            total += f[0]
            count += 1
        return total/count

    def next_gen(self, per_p, per_s, per_k, *args):
        """Generate the next generation"""
        # decide how many survivors and parents to keep
        assert per_p + per_s <= 1
        sorted_fitness = sorted(self.fitness)
        kept_idx = math.floor(per_p * len(self.individuals))
        survivor_idx = math.floor(per_s * len(self.individuals))
        parents = []
        for i in range(kept_idx):
            parent_idx = sorted_fitness[i][1]
            parents.append(self.individuals[parent_idx].copy())

        # keep a percentage of non parents to help avoid local minimum
        survivor = []
        count = 0
        while len(survivor) < survivor_idx:
            idx_s = random.randint(0, self.n - 1)
            selected = self.individuals[idx_s].copy()
            if per_k < random.uniform(0,1) and not (selected in parents):
                survivor.append(selected)
            count += 1
            if count > 1e5:  # if stuck in the loop because all same, new indv (migration?)
                survivor.append(self.indv_gen(*args))

        # find the number of children required
        n_children = len(self.individuals) - kept_idx - survivor_idx
        children = self.produce_children(parents, n_children)
        self.individuals = parents + children + survivor

    def produce_children(self, parents, n_children):
        children = []
        n_parents = len(parents)
        n_params = len(parents[0])
        assert n_parents >= 2
        for i in range(n_children):
            # pick the parents to use; ensure they aren't the same
            p1, p2 = None, None
            while p1 == p2:
                p1, p2 = random.randint(0, n_parents-1), random.randint(0, n_parents-1)

            # for a given child, take half of parent 1 and half of parent 2
            child = []
            for i in range(n_params):
                if i % 2 == 0:
                    child.append(parents[p1][i])
                else:
                    child.append(parents[p2][i])
            children.append(child)

        return children

    def mutate(self, p_mut, *args):
        for ind in self.individuals:
            if random.uniform(0,1) < p_mut:
                mut_idx = random.randint(0,len(ind)-1)
                ind[mut_idx] = self.mut_func(*args)

    def best_fit(self):
        idx_best = sorted(self.fitness)[0][1]
        return self.individuals[idx_best]






#-------------------------------Student Problem--------------------------------#
def logistic_function(beta, h):
    # logistic function for logistic regression
    return 1/(1 + math.exp(-(beta[0] + beta[1]*h)))


def loss_function(beta, hour, passed):
    # square error loss function; used to measure fitness
    sse = 0
    for h,p in zip(hour,passed):
        sse += (p - logistic_function(beta, h))**2
    return sse


def generate_individual(a, b, n):
    # generates a list of n random parameters between a and b
    return [random.uniform(a,b) for i in range(n)]

def mutate(a,b):
    # how a mutation on a parameter can be definied
    return random.uniform(a,b)

#------------------------------------Data--------------------------------------#
hours = [ 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
        2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50 ]

passed = [ 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 1, 1, 1, 1, 1]

if __name__ == "__main__":
    # initialize parameters
    p_size = 40
    param_bounds = [-10, 10]
    n_params = 2
    per_p = 0.3
    per_k = 0.5
    per_s = 0.1
    p_mute = 0.05
    hours_args = [hours for i in range(p_size)]
    passed_args = [passed for i in range(p_size)]

    # begin optimizations
    p = Population(p_size, loss_function, generate_individual, mutate)
    p.generate_population(*param_bounds, n_params)
    for i in range(1000):
        p.eval_pop_fitness(hours_args, passed_args)
        p.next_gen(per_p, per_s, per_k, *param_bounds, n_params)
        p.mutate(p_mute,*param_bounds)
        print("Average fitness: ", p.average_fitness())
    print("Final Parameters: ", p.best_fit())
