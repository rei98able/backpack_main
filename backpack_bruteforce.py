import pandas as pd
from itertools import combinations
from time import time
from numpy.testing import assert_allclose

tests = ["5", "8", "10", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26"]
# tests = ["5", "8"]


def backpack_brute_force(n, capacity, weight_cost):
    best_cost = None
    best_combination = []
    # generating combinations : C by 1 from n, C by 2 from n, ...
    # combinations(range(4), 3) --> 012 013 023 123
    for i in range(n):
        for combination in combinations(weight_cost, i + 1):
            weights = sum([weight[0] for weight in combination])
            costs = sum([cost[1] for cost in combination])
            if (best_cost is None or best_cost < costs) and weights <= capacity:
                best_cost = costs
                best_combination = [0] * n
                for c in combination:
                    best_combination[weight_cost.index(c)] = 1
    return best_cost, best_combination


def main():
    for file in tests[:10]:
        test = pd.read_csv("test_" + file + ".csv", header=None, delimiter=";").values
        result = pd.read_csv("bpresult_" + file + ".csv", header=None, delimiter=";").values.ravel()
        n = test.shape[0]
        capacity = sum(test[:, 0]) / 2
        tuple_array = [tuple(x for x in row) for row in test]
        # print(tuple_array, capacity)
        best_cost, best_combination = backpack_brute_force(n, capacity, tuple_array)
        # print(assert_allclose(best_combination, result))
        print(best_combination, file)
        # backpack_brute_force(n, capacity, tuple_array)


start = time()
main()
end = time()
delta = end - start
print("Time: ", delta)
