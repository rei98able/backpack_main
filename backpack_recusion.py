import pandas as pd
from time import time

from numpy.testing import assert_allclose

tests = ["5", "8", "10", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26"]
# tests = ["5", "8"]


def best_value(i, j, weight_cost):
    if i == 0:
        return 0
    weight, cost = weight_cost[i - 1]
    if weight > j:
        return best_value(i - 1, j, weight_cost)
    else:
        # maximizing the cost
        return max(best_value(i - 1, j, weight_cost), best_value(i - 1, j - weight, weight_cost) + cost)


def dynamic_programming(number, capacity, weight_cost):

    j = capacity
    result = [0] * number
    for i in range(len(weight_cost), 0, -1):
        if best_value(i, j, weight_cost) != best_value(i - 1, j, weight_cost):
            result[i - 1] = 1
            j -= weight_cost[i - 1][0]
    return best_value(len(weight_cost), capacity, weight_cost), result


def main():
    for file in tests[:]:
        test = pd.read_csv("test_" + file + ".csv", header=None, delimiter=";").values
        result = pd.read_csv("bpresult_" + file + ".csv", header=None, delimiter=";").values.ravel()
        n = test.shape[0]
        capacity = sum(test[:, 0]) / 2
        tuple_array = [tuple(x for x in row) for row in test]
        # print(tuple_array, capacity)
        best_cost, best_combination = dynamic_programming(n, capacity, tuple_array)
        # print(assert_allclose(best_combination, result))
        print(best_cost)
        print(best_combination)
        # backpack_brute_force(n, capacity, tuple_array)


start = time()
main()
end = time()
delta = end - start
print("Time: ", delta)
