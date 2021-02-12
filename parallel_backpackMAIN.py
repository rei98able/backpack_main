import pandas as pd
import numpy as np
from time import time
from numpy.testing import assert_allclose
from joblib import Parallel, delayed
from random import uniform
import matplotlib.pyplot as plt

tests = ['5', '8', '10', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']


# tests = ["5", "8"]

def generate_tests(num):
    for num_tests in range(27, num + 1):
        f = open('test_' + str(num_tests) + '.csv', 'w', encoding='utf-8')
        for _ in range(num_tests):
            f.write(str(uniform(10, 1000)) + ';' + str(uniform(10, 1000)) + '\n')
        f.close()
    return [str(i) for i in range(27, num)]


def dynamic_parallel_backpack(args):
    capacity = args[0]
    weights = args[1]
    costs = args[2]
    n = args[3]
    capacity = int(capacity)
    best_costs = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    best_combination = [0] * n
    for i in range(n + 1):
        for j in range(capacity + 1):
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


result_files = []
time_arr = []
acc_arr = []
for file in tests:
    results = pd.read_csv("bpresult_" + file + ".csv", header=None, delimiter=";").values.ravel()
    result_files.append(results)


tests2 = generate_tests(100)
for test in tests2:
    tests.append(test)
test_files = []
for file in tests:
    test = pd.read_csv("test_" + file + ".csv", header=None, delimiter=";").values
    n = test.shape[0]
    cap = sum(test[:, 0]) / 2
    test_files.append([cap, test[:, 0], test[:, 1], n])

for job in range(1, 9):
    result_a = []
    start = time()
    result = Parallel(n_jobs=job)(delayed(dynamic_parallel_backpack)(d) for d in test_files)
    delta = time() - start
    time_arr.append(delta)
    for j in result:
        result_a.append(j[1])

one = time_arr[0]
for time in range(1, len(time_arr)):
    acc_arr.append(one / time_arr[time])

for i in range(len(result_files)):
    if not assert_allclose(result_files[i], result_a[i]):
        print("OK")

print(time_arr, '\n', acc_arr)

# plt.plot(np.arange(1, 9), time_arr)
plt.plot(np.arange(2, 9), acc_arr)
plt.xlabel('Количество процессов')
plt.ylabel('Ускорение в x раз')
plt.show()
