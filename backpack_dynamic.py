import pandas as pd
from time import time


tests = ["5", "8", "10", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26"]
# tests = ["5", "8"]


def dynamic_parallel_backpack(args):
    capacity = args[0]
    weights = args[1]
    costs = args[2]
    n = args[3]
    capacity = int(capacity)
    best_costs = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    best_combination = [0] * n
    for i in range(n + 1):
        for j in range(int(capacity + 1)):
            if i == 0 or j == 0:
                best_costs[i][j] = 0
            elif weights[i - 1] <= j:
                best_costs[i][j] = max(costs[i - 1] + best_costs[i - 1][j - int(weights[i - 1])], best_costs[i - 1][j])
            else:
                best_costs[i][j] = best_costs[i - 1][j]

    result = best_costs[n][capacity]
    cap_copy = capacity
    for i in range(n, 0, -1):
        if round(result) != round(best_costs[i - 1][int(cap_copy)]):
            best_combination[i - 1] = 1
            result -= costs[i - 1]
            cap_copy -= weights[i - 1]
    return best_costs[n][capacity], best_combination


if __name__ == '__main__':
    start = time()
    for file in tests:
        test = pd.read_csv("test_" + file + ".csv", header=None, delimiter=";").values
        results = pd.read_csv("bpresult_" + file + ".csv", header=None, delimiter=";").values.ravel()
        n = test.shape[0]
        cap = sum(test[:, 0]) / 2
        # print(tuple_array, capacity)
        best_сost, best_combination = dynamic_parallel_backpack([cap, test[:, 0], test[:, 1], n])
        print(best_combination)
    end = time()
    delta = end - start
    print("Time: ", delta)


