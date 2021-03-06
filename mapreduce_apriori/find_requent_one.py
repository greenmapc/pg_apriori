import multiprocessing
import timeit
from multiprocessing import Process


def find_frequent_one_simple(data_set, support):
    candidate_one = {}
    total = len(data_set)
    for row in data_set.values():
        for val in row:
            if val in candidate_one:
                candidate_one[val] += 1
            else:
                candidate_one[val] = 1

    frequent_1 = []
    for key, cnt in candidate_one.items():
        # check if given item has sufficient count.
        if cnt >= (support * total / 100):
            frequent_1.append(([key], cnt))
    return frequent_1


def find_frequent_one(dataset, support_cnt):

    def separate_data_for_processes(processes_size, dataset):
        separate_start = timeit.default_timer()
        datasets = []
        step = int(len(dataset) / processes_size)
        border = step
        current_data = {}
        item_num = 0
        last_chunk = False
        for i in dataset.items():
            if (item_num == border and not last_chunk) or item_num == len(dataset) - 1:
                datasets.append(current_data)
                if last_chunk:
                    current_data[i[0]] = i[1]
                if len(datasets) == processes_size - 1:
                    last_chunk = True
                current_data = {}
                border += step
            current_data[i[0]] = i[1]
            item_num += 1
        separate_stop = timeit.default_timer()
        print("separate dataset time ", separate_stop - separate_start)
        return datasets

    def find_frequent_one_map(dataset, map_result):
        result = {}
        for transaction in dataset.values():
            for item in transaction:
                if item in result:
                    result[item] += 1
                else:
                    result[item] = 1
        map_result.put(result)

    def run_find_frequent_map(processes_size, separated_dataset):
        map_start = timeit.default_timer()
        map_result = multiprocessing.Manager().Queue()
        jobs = []
        for i in range(processes_size):
            j = Process(target=find_frequent_one_map,
                        args=(separated_dataset[i], map_result))
            jobs.append(j)
            j.start()

        for job in jobs:
            job.join()

        map_stop = timeit.default_timer()
        print("map time = ", map_stop - map_start)
        return map_result

    def find_frequent_one_shuffle(map_result):
        shuffle_start = timeit.default_timer()
        shuffle_result = dict()
        while not map_result.empty():
            current = map_result.get()
            for key, value in current.items():
                if key in shuffle_result:
                    shuffle_result[key].append(value)
                else:
                    shuffle_result[key] = []
                    shuffle_result[key].append(value)
        shuffle_stop = timeit.default_timer()
        print('shuffle time', shuffle_stop - shuffle_start)
        return shuffle_result

    def find_frequent_one_reduce(map_result, min_support, reduce_result):
        result = dict()
        for key, value in map_result.items():
            current_count = 0
            for term in value:
                current_count += term
            if current_count >= min_support:
                result[key] = current_count
        reduce_result.put(result)

    def run_find_frequent_reduce(processes_size, shuffle_result, min_support):
        reduce_start = timeit.default_timer()
        separated_dataset = separate_data_for_processes(processes_size, shuffle_result)

        reduce_result = multiprocessing.Manager().Queue()
        jobs = []
        for i in range(processes_size):
            j = Process(target=find_frequent_one_reduce,
                        args=(separated_dataset[i], min_support, reduce_result))
            jobs.append(j)
            j.start()

        for job in jobs:
            job.join()

        reduce_stop = timeit.default_timer()
        print("reduce time ", reduce_stop - reduce_start)
        return reduce_result

    processes_size = 2
    separated_dataset = separate_data_for_processes(processes_size, dataset)
    start = timeit.default_timer()
    map_result = run_find_frequent_map(processes_size, separated_dataset)
    print("map function finished")
    shuffle_result = find_frequent_one_shuffle(map_result)
    print("shuffle function finished")
    support_cnt = support_cnt * len(dataset) / 100
    reduce_result = run_find_frequent_reduce(processes_size, shuffle_result, support_cnt)
    print("reduce function finished")
    check_sum = 0
    result = []
    stop = timeit.default_timer()
    print("MAP REDUCE TIME", stop - start)
    while not reduce_result.empty():
        current = reduce_result.get()
        for item, val in current.items():
            result.append(([item], val))
            check_sum += val
        # print(current)
    # print(check_sum)
    return result


def find_frequent_one_simple_run(dataset, support):
    start = timeit.default_timer()
    res = find_frequent_one_simple(dataset, support)
    stop = timeit.default_timer()
    print("SIMPLE ALGORITHM TIME ", stop - start)
    return res
