#! /usr/bin/env python
"""
A simple genetic algorithm class meant to solve general optimization problems

author: edelsonc
date: 7/14/2017
"""
import math
import random

class Population(object):
    """
    Generic population class for genetic algorithms. This class is intended to
    be used for optimization problems whre a loss function can be clearly
    defined. As such, the class requires some basic parameters

    n -- number of individuals in the population
    loss_func -- loss function for the optimization problem. `Population`
        expects minimization problems. If you're maximizing, you can fix this
        by substituting with `lambda x: -my_loss_function(x)`. The loss
        function should also accept arguments in the following order
        `loss_func(individual, y, X, *args)` where y is the result vector and
        X is the data matrix.
    indv_gen -- an individual generation function; can accept *args
    mut_func -- a mutation function; will operate on an individual

    For the purpose of the `Population`, and individual is considered a single
    set of parameters. See `example.py` for the case of a logistic regression.
    """
    def __init__(self, n, loss_func, indv_gen, mut_func):
        "instantiated variables; arguments described in class docstring"
        self.individuals = []
        self.loss_func = loss_func
        self.indv_gen = indv_gen
        self.mut_func = mut_func
        self.n = n
        self.y, self.X = None, None
        self.fitness = None

        

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
        # check to make sure training data is set
        assert self.X != None
        assert self.y != None
        
        if self.fitness == None:
            self.fitness = []
            for i, indv in enumerate(self.individuals):
                self.fitness.append( (self.loss_func(indv, self.y, self.X, *args), i) )
        else:
            for i, indv in enumerate(self.individuals):
                self.fitness[i] = (self.loss_func(indv, self.y, self.X, *args), i)

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
        assert per_p + per_s <= 1

        # decide how many survivors and parents to keep
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
        """generates children from random parents"""
        children = []
        n_parents = len(parents)
        n_params = len(parents[0])

        # we're not unicellular here!
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
        """selects the individual with the best fitness score"""
        idx_best = sorted(self.fitness)[0][1]
        return self.individuals[idx_best]

    def set_train(self, y, X):
        """set function for setting the trainig data"""
        self.y = y
        self.X = X
