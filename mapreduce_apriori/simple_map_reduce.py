import multiprocessing
from multiprocessing import Pool


def if_prime(x):
    print(x, multiprocessing.current_process().pid)
    if x <= 1:
        return 0
    elif x <= 3:
        return x
    elif x % 2 == 0 or x % 3 == 0:
        return 0
    i = 5
    while i ** 2 <= x:
        if x % i == 0 or x % (i + 2) == 0:
            return 0
        i += 6
    return x


answer = 0

if __name__ == '__main__':
    with Pool(2) as p:
        res = p.map(if_prime, list(range(1000)))
        answer = sum(res)
    print(answer)
