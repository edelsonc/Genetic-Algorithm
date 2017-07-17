#! /usr/bin/env python
"""
Example using the Population calss of the genetic_algorithm module. This is the
well known classification problem of students passing an exam based on the
number of hours spend studying. To solve this problem here, logistic regression
is used.

author: edelsonc
created: 07/16/17
"""
import math
import random
from genetic_algorithm import Population

#-------------------------------Student Problem--------------------------------#
def logistic_function(beta, h):
    """logistic function for logistic regression"""
    return 1/(1 + math.exp(-(beta[0] + beta[1]*h)))


def loss_function(beta, passed, hour):
    """square error loss function; used to measure fitness"""
    sse = 0
    for h,p in zip(hour,passed):
        sse += (p - logistic_function(beta, h))**2
    return sse


def generate_individual(a, b, n):
    """Generates a list of n random parameters between a and b"""
    return [random.uniform(a,b) for i in range(n)]

def mutate(a,b):
    """how a mutation on a parameter can be definied"""
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
    hours_args = hours # [hours for i in range(p_size)]
    passed_args = passed # [passed for i in range(p_size)]

    # begin optimizations
    p = Population(p_size, loss_function, generate_individual, mutate)
    p.generate_population(*param_bounds, n_params)
    p.set_train(passed, hours)
    for i in range(1000):
        p.eval_pop_fitness() #hours_args, passed_args)
        p.next_gen(per_p, per_s, per_k, *param_bounds, n_params)
        p.mutate(p_mute,*param_bounds)
        print("Average fitness: ", p.average_fitness())
    print("Final Parameters: ", p.best_fit())
