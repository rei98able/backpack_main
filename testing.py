import multiprocessing as mp
import time
import pandas as pd
from math import sqrt, inf


def find_price(values_, n_, capacity_):
    current = 1
    w = 0.0
    price = 0.0
    for v in values_:
        if (n_ & current) != 0:
            w += v[0]
            price += v[1]
            if w >= capacity_:
                break
        current <<= 1
    return price

def task(args):
    pid = args[0]
    num_threads = args[1]
    values = args[2]
    capacity = args[3]
    N = values.shape[0]
    S = pow(2, N)
    s = S // num_threads
    start = s * pid
    count = s if (pid != (num_threads - 1)) else s + S % num_threads
    result = 0
    max_price = 0
    for i in range(start, start + count):
        p = find_price(values, i, capacity)
        if p > max_price:
            result = i
            max_price = p
    return result, max_price


tests = ["5", "8", "10", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26"]


for t in tests[:]:
    if __name__ == '__main__':
        test = pd.read_csv("test_" + t + ".csv", header=None, delimiter=";").values
        N = test.shape[0]
        num_threads = 16

        capacity = sum(test[:, 0]) / 2
        with mp.Pool(num_threads) as p:
            res = p.map(task, [(i, num_threads, test, capacity) for i in range(num_threads)])
            res = sorted(res, key=lambda x: x[1], reverse=True)
            file = open("bpresult_" + t + ".csv", "w")
            current = 1
            for v in test:
                if current != 1:
                    file.write(";")
                if (current & res[0][0]) != 0:
                    file.write("1")
                else:
                    file.write("0")
                current <<= 1
            file.write("\n")
            print(res)