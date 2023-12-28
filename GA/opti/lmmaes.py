import numpy as np
import math
import sys
import time

def l2norm(vec):
    return vec / np.sqrt(np.sum(vec ** 2, -1, keepdims=True))


class Lmmaes:
    """
    Implementation of the Limited-Memory Matrix Adaptation Evolution Strategy (LM-MA-ES) proposed in the following paper:

    Large Scale Black-box Optimization by Limited-Memory Matrix Adaptation. Ilya Loshchilov, Tobias Glasmachers, and Hans-Georg Beyer, IEEE Transactions on Evolutionary Computation 99, 2018.

    The class provides an ask-and-tell interface to the optimizer. In each generation, the user *asks* for a set of points (the current population) and then *tells* the class about their objective function values.
    Alternatively, a run method is provided, which takes the evaluation procedure as a function argument.

    Algorithm by Ilya Loshchilov. Python implementation by Tobias Glasmachers. Refactoring, packaging and distribution by Giuseppe Cuccu. All rights reserved.
    """
    def __init__(self,
            # x,                # Initial individual (My modification)
            fit_object,       # Fitness function (My modification)
            number_of_generations, #(my modification)
            sigma,            # Initial step size
            popsize=None,     # Custom population size
            rseed=None,       # Set random seed for reproducibility
            verbose=False,     # Verbose printing to stdout
            print_interval=1, # Printout interval - only if verbose

        ):
        """
        Construct a new solver instance. You need to provide at least an
        initial search point (x) and an initial step size (sigma).
        """
        self.number_of_generations = number_of_generations

        if fit_object.data.shape[0] != 0:  # if p_data is not empty:
            x = np.mean(fit_object.data, axis=0)
        else:
            x = -np.mean(fit_object.n_data, axis=0)  # if its empty, lets set the initial point as the negated average of n-set, to be as far as possisble

        self.fit_object = fit_object
        self.init_ind = x # saving it uniquely for debugging purposes
        self.rseed = rseed
        if self.rseed is not None:
            np.random.seed(self.rseed)
        self.verbose = verbose

        # preparation
        self.x = x.copy() # make sure it's writeable and separated from init_ind
        self.sigma = sigma
        self.n = len(x)
        self.m = popsize or 2*int(4 + math.floor(3 * math.log(self.n)))  #when number of features is 512, the population will be 44
        self.lbd = self.m
        self.mu = self.lbd // 2
        assert self.lbd < self.n, 'Tuning constants break, use CMA-ES instead'
        self.best_ind = None
        self.best_fit = float('inf') # minimization!

        # run init
        self.start_time = time.time()
        self.t = 0   # number of generations maintained by ask/tell
        self.w = [math.log(self.lbd/2 + 0.5) - math.log(i + 1) for i in range(self.mu)]
        self.w /= np.sum(self.w)
        mu_w = 1 / np.sum(np.square(self.w))
        self.p_sigma = np.zeros((self.n,))

        self.c_sigma = 2 * self.lbd / self.n
        self.c_sigma_update = math.sqrt(mu_w * self.c_sigma * (2 - self.c_sigma))
        self.c_d = np.zeros((self.m,))
        self.c_c = np.zeros((self.m,))
        self.c_c_update = np.zeros((self.m,))
        for i in range(self.m):
            self.c_d[i] = 1 / (math.pow(1.5, i) * self.n)
            self.c_c[i] = self.lbd / (math.pow(4.0, i) * self.n)
            self.c_c_update[i] = math.sqrt(mu_w * self.c_c[i] * (2 - self.c_c[i]))
        self.d_sigma = 1.0
        self.chi = math.sqrt(self.n) * (1 - 1/(4*self.n) - 1/(21*self.n*self.n));
        self.M = np.zeros((self.m, self.n))

        if self.verbose:
            print(f"[LM-MA-ES]  n={self.n}  m={self.m}  " + \
                  f"mu={self.mu}  lambda={self.lbd}")


        self.temp_values = np.zeros((self.lbd,))



    @property
    def popsize(self):
        """Syntactic sugar: return the population size."""
        return self.m

    @property
    def f_evaluations(self):
        """Return the number of objective-function evaluations so far."""
        return self.lbd * self.t

    @property
    def runtime(self):
        """Return the (wall clock) runtime since the construction of the solver."""
        return time.time() - self.start_time

    def report(self, multi_line=False):
        """Report progress to standard output"""
        if not self.verbose: return
        # if self.t % self.print_interval != 0: return
        delete = '' if multi_line else '\b'*100
        end = '\n' if multi_line else ''
        bud = f"#FE: {self.lbd * self.t}  "
        gen = f"gen: {self.t:,}"
        fit = f"fit: {self.best_fit:.3e}  "
        sig = f"step size: {self.sigma:.3e}  "
        print(delete, ' '*3, bud, gen, fit, sig, delete, end=end)


    def ask(self):
        """
        Start the next generation by sampling an offspring
        population of points. The points are returned as a
        2D numpy array, representing a list of points for
        (external) evaluation. See *tell* for further details.
        """
        # sample offspring, vectorized version
        self.z = np.random.randn(self.lbd, self.n)
        self.d = np.copy(self.z)
        for j in range(min(self.t, self.m)):
            self.d = (1 - self.c_d[j]) * self.d + \
                self.c_d[j] * np.outer(np.dot(self.d, self.M[j,:]), self.M[j,:])
        self.population = self.x + self.sigma * self.d
        self.population = l2norm(self.population)
        return self.population


    def tell(self, values):
        """
        Each call to *ask* must be followed by a corresponding call to *tell*.
        The function expects a 1D numpy array containing the objective function
        values of the points returned by ask, in the same order.
        """

        # sort by fitness
        order = np.argsort(values)

        # maintain best ind
        cur_best_fit = values[order[0]]
        if cur_best_fit < self.best_fit: # minimization!
            self.best_fit = cur_best_fit
            self.best_ind = self.population[order[0],:]

        # update mean
        for i in range(self.mu):
            self.x += self.sigma * self.w[i] * self.d[order[i],:]

        # compute weighted mean
        wz = np.zeros((self.n,))
        for i in range(self.mu):
            wz += self.w[i] * self.z[order[i],:]

        # update evolution path
        self.p_sigma *= 1 - self.c_sigma
        self.p_sigma += self.c_sigma_update * wz

        # update direction vectors
        for i in range(self.m):
            self.M[i,:] *= 1 - self.c_c[i]
            self.M[i,:] += self.c_c_update[i] * wz

        # update step size
        self.sigma *= math.exp(self.c_sigma / self.d_sigma *\
                               (np.linalg.norm(self.p_sigma) / self.chi - 1))

        # update generation number
        self.t += 1

    def run(self, f_target=None, budget=None, generations=None):
        """
        Run LM-MA-ES to minimize an objective function fit_fn.
        At least one of the following stopping criteria must be specified:
         * budget: maximal number of objective function evaluations,
         * generations: maximal number of generations to run,
         * f_target: target objective function value.
        """

        assert budget is not None or f_target is not None or generations is not None
        if generations is not None:
        	if budget is None:
        		budget = self.lbd * generations
        	else:
        		budget = max(budget, self.lbd * generations)

        if budget is not None:
	        budget += self.f_evaluations;   # continuing a run is fine

        # generation loop
        counter = 0
        while True:
            if budget is not None and self.f_evaluations >= budget: break
            if f_target is not None and self.best_fit <= f_target: break

            self.step()

            #my modification:
            if counter % 10 == 0:
                print(counter ,'sum of dist: ',  self.fit_object.sum_of_dist(self.best_ind))
                print(counter, 'count of lockeds: ', self.fit_object.count_of_locks(self.best_ind))
            #:my modification
            counter+=1

            # end-of-generation printout
            # self.report()
            # print(self.best_fit)
        print(f"\nOptimization run stopped after {self.f_evaluations} objective function evaluations with final value {self.best_fit} ({self.runtime} seconds)")
        return self.best_ind, self.f_evaluations


    def step(self):
        # ask() for a new population
        population = self.ask()

        # evaluate offspring
        for i in range(self.lbd):
            self.temp_values[i] = self.fit_object.evaluate(population[i])

        # tell() fitnesses to update
        self.tell(self.temp_values)



