import numpy as np
from .problem import Problem


def euclidean(a, b):
    return np.linalg.norm(a-b)
def cosine(a,b):
    return 1 - np.sum(a*b)/(np.linalg.norm(a) * np.linalg.norm(b))

def l2norm(vec):
    return vec / np.sqrt(np.sum(vec ** 2, -1, keepdims=True))

import math
def cosine_variant(a, b):
    dot = np.sum(np.multiply(a, b))
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    similarity = dot / norm
    return np.arccos(similarity) / math.pi

class LockAndDistance(Problem):
    def __init__(self, data, threshold, dist_function):
        self.dist_function = dist_function
        self.threshold = threshold
        self.data = data.reshape(-1, data.shape[-1])
        super().__init__(dim=data.shape[-1], lb=None, up=None)

    # def __init__(self,dataset, lb=None, ub=None, threshold=1.04):
    #     self.threshold = threshold
    #     loaded = np.load('../datasets/'+ dataset)
    #     data = loaded['X_embedding']
    #     self.data = data.reshape(-1, data.shape[-1])
    #     super().__init__(dim=data.shape[-1], lb=lb, up=ub)

    def sum_of_dist(self, x):
        return sum(self.dist_function(x, d) for d in self.data)

    def count_of_locks(self, x):
        return sum(self.dist_function(x, d) > self.threshold for d in self.data)


    def evaluate(self, x):
        pass


class DistanceSum(LockAndDistance):

    def __init__(self,data, threshold, dist_function):
        super().__init__(data, threshold=threshold, dist_function=dist_function)

    def evaluate(self, x):
        return self.sum_of_dist(x)


class LockCount(LockAndDistance):
    def __init__(self,data, threshold, dist_function):
        super().__init__(data, threshold=threshold, dist_function=dist_function)

    def evaluate(self, x):
        return self.count_of_locks(x)

class LockCountAndDistanceSum(LockAndDistance):

    def __init__(self,data, threshold, dist_function):
        super().__init__(data, threshold=threshold, dist_function=dist_function)

    def evaluate(self, x):
        re = self.count_of_locks(x) + self.sum_of_dist(x)
        return re



class Weighted_PSet_NSet_LockAndDistance(Problem):
    def __init__(self,
                 p_data, p_data_lock_weight,
                 n_data, n_data_lock_weight,
                 p_n_balance_weight,
                 threshold_for_p,
                 threshold_for_n,
                 dist_function):
        assert 0 <= p_data_lock_weight <= 1 , 'weights should be of [0,1]'
        assert 0 <= n_data_lock_weight <= 1 , 'weights should be of [0,1]'
        assert 0 <= p_n_balance_weight <= 1 , 'weights should be of [0,1]'

        self.p_n_balance_weight = p_n_balance_weight
        self.n_data_lock_weight = n_data_lock_weight
        self.p_data_lock_weight = p_data_lock_weight
        self.dist_function = dist_function
        self.threshold_for_p = threshold_for_p
        self.threshold_for_n = threshold_for_n

        if p_data.shape[0] != 0:  # if p_data is not empty:
            self.data = self.p_data = p_data.reshape(-1, p_data.shape[-1])
        else:
            self.data = self.p_data = p_data

        if n_data.shape[0] != 0: # if n_data is not empty:
            self.n_data = n_data.reshape(-1, n_data.shape[-1])
        else:
            self.n_data = n_data
        super().__init__(dim=p_data.shape[-1], lb=None, up=None)

    def sum_of_dist(self, point, points):
        return sum(self.dist_function(point, d) for d in points)

    def sum_of_dist_from_p(self, x):
        return self.sum_of_dist(x, self.p_data)

    def sum_of_dist_from_n(self, x):
        return self.sum_of_dist(x, self.n_data)

    def count_of_locks(self, point, points, threshold):
        return sum(self.dist_function(point, d) > threshold for d in points)

    def count_of_p_locks(self, x):
        return self.count_of_locks(x, self.p_data, self.threshold_for_p)

    def count_of_n_locks(self, x):
        return self.count_of_locks(x, self.n_data, self.threshold_for_n)


    def evaluate(self, x):
        if self.p_data.shape[0] != 0: # if p_data is not empty:
            p_loss = self.p_data_lock_weight * self.count_of_p_locks(x) + (1 - self.p_data_lock_weight) * self.sum_of_dist_from_p(x)
            p_loss = p_loss / self.p_data.shape[0]
        else:
            p_loss = 0

        if self.n_data.shape[0] != 0: # if n_data is not empty:
            n_loss = self.n_data_lock_weight * self.count_of_n_locks(x) + (1 - self.n_data_lock_weight) * self.sum_of_dist_from_n(x)
            n_loss = n_loss / self.n_data.shape[0]
        else:
            n_loss = 0

        loss = self.p_n_balance_weight * p_loss - (1 - self.p_n_balance_weight) * n_loss

        return loss


class WeightedLockCountAndDistanceSum(LockAndDistance):

    def __init__(self,data,  threshold, lock_weight, dist_function):
        super().__init__(data, threshold=threshold, dist_function=dist_function)
        self.lock_weight = lock_weight

    def evaluate(self, x):
        re = self.lock_weight * self.count_of_locks(x) + self.sum_of_dist(x)
        return re

class LockCountAndDistanceSumSmart(LockAndDistance):

    def __init__(self,data, threshold, dist_function, threshold_for_sum_of_dists=2600):
        super().__init__(data, threshold=threshold, dist_function=dist_function)
        self.threshold_for_sum_of_dists = threshold_for_sum_of_dists

    def evaluate(self, x):
        re = self.sum_of_dist(x)
        if re < self.threshold_for_sum_of_dists:
            re = self.count_of_locks(x)

        return re














#
# class Sphere(Problem):
#
#     def __init__(self, dim, lb, ub):
#         super().__init__(dim, lb, ub)
#
#     def evaluate(self, x):
#         x = self.lb + x * (self.ub - self.lb)
#         re = sum(np.power(x[i], 2) for i in range(self.D))
#         return re
#
#
# class Rosenbrock(Problem):
#
#     def __init__(self, dim, lb, ub):
#         super().__init__(dim, lb, ub)
#
#     def evaluate(self, x):
#         x = self.lb + x * (self.ub - self.lb)
#         re = 0
#         for i in range(self.D - 1):
#             re += 100 * np.power((np.power(x[i], 2) - x[i + 1]), 2) + np.power((x[i] - 1), 2)
#         return re
#
#
# class Ackley(Problem):
#
#     def __init__(self, dim, lb, ub):
#         super().__init__(dim, lb, ub)
#
#     def evaluate(self, x):
#         x = self.lb + x * (self.ub - self.lb)
#
#         # shift operation
#         # x = x - 42.0969
#
#         part1 = 0
#         for i in range(self.D):
#             part1 += np.power(x[i], 2)
#         part2 = 0
#         for i in range(self.D):
#             part2 += np.cos(2 * np.pi * x[i])
#         re = -20 * np.exp(-0.2 * np.sqrt(part1 / self.D)) \
#             - np.exp(part2 / self.D) + 20 + np.e
#         return re
#
#
# class Rastrgin(Problem):
#
#     def __init__(self, dim, lb, ub):
#         super().__init__(dim, lb, ub)
#
#     def evaluate(self, x):
#         x = self.lb + x * (self.ub - self.lb)
#         re = 0
#         for i in range(self.D):
#             re += x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i]) + 10
#         return re
#
#
# class Griewank(Problem):
#
#     def __init__(self, dim, lb, ub):
#         super().__init__(dim, lb, ub)
#
#     def evaluate(self, x):
#         x = self.lb + x * (self.ub - self.lb)
#         part1, part2 = 0, 1
#         for i in range(self.D):
#             part1 += x[i] ** 2
#             part2 *= np.cos(x[i] / np.sqrt(i + 1))
#         re = 1 + part1 / 4000 - part2
#         return re
#
#
# class Weierstrass(Problem):
#
#     def __init__(self, dim, lb, ub):
#         super().__init__(dim, lb, ub)
#
#     def evaluate(self, x):
#         x = self.lb + x * (self.ub - self.lb)
#         part1 = 0
#         for i in range(self.D):
#             for j in range(21):
#                 part1 += np.power(0.5, j) * np.cos(2 * np.pi * np.power(3, j) * (x[i] + 0.5))
#         part2 = 0
#         for i in range(21):
#             part2 += np.power(0.5, i) * np.cos(2 * np.pi * np.power(3, i) * 0.5)
#         re = part1 - self.D * part2
#         return re
#
#
# class Schwefel(Problem):
#
#     def __init__(self, dim, lb, ub):
#         super().__init__(dim, lb, ub)
#
#     def evaluate(self, x):
#         x = self.lb + x * (self.ub - self.lb)
#         part1 = 0
#         for i in range(self.D):
#             part1 += x[i] * np.sin(np.sqrt(np.abs(x[i])))
#         re = 418.9829 * self.D - part1
#         return re
