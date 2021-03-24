import multiprocessing
import timeit
from multiprocessing import Process

from mapreduce_apriori.sep.trie import binary_search


def separate_data_for_processes(processes_size, dataset):
    separate_start = timeit.default_timer()
    datasets = []
    step = len(dataset) // processes_size
    border = step
    current_data = {}
    item_num = 0
    if len(dataset) <= border:
        last_chunk = True
    else:
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
    if current_data:
        datasets.append(current_data)
    separate_stop = timeit.default_timer()
    print("Separate dataset time ", separate_stop - separate_start)

    return datasets


def shuffle_function(map_result):
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
    print('Shuffle time', shuffle_stop - shuffle_start)
    return shuffle_result


def reduce_function(processes_size, shuffle_result, min_support):
    def reduce(map_result, min_support, reduce_result):
        result = dict()
        for key, value in map_result.items():
            current_count = 0
            for term in value:
                current_count += term
            if current_count >= min_support:
                result[key] = current_count
        reduce_result.put(result)

    reduce_start = timeit.default_timer()
    separated_dataset = separate_data_for_processes(processes_size, shuffle_result)
    print(separated_dataset)
    print(shuffle_result)

    reduce_result = multiprocessing.Manager().Queue()
    jobs = []
    for i in range(processes_size):
        j = Process(target=reduce,
                    args=(separated_dataset[i], min_support, reduce_result))
        jobs.append(j)
        j.start()

    for job in jobs:
        job.join()

    reduce_stop = timeit.default_timer()
    print("reduce time ", reduce_stop - reduce_start)
    return reduce_result


def convert_reduce_result(reduce_result):
    result = []
    while not reduce_result.empty():
        current = reduce_result.get()
        for item, val in current.items():
            result.append(([item], val))
    return result


def find_frequent_one(dataset, support_cnt):
    def map(dataset, map_result, left_border, right_border):
        result = {}
        for i in range(left_border, right_border):
            for item in dataset[i]:
                if item in result:
                    result[item] += 1
                else:
                    result[item] = 1
        map_result.put(result)

    def find_frequent_map(processes_size, dataset):
        map_start = timeit.default_timer()
        map_result = multiprocessing.Manager().Queue()
        jobs = []
        left_border = 0
        dataset_len = len(dataset)
        step = int(dataset_len / processes_size)
        right_border = step
        for i in range(processes_size):
            j = Process(target=map,
                        args=(dataset, map_result, left_border, right_border))
            print("run map with left " + str(left_border) + " and right " + str(right_border))
            left_border += step
            if j == processes_size - 1:
                right_border = dataset_len
            else:
                right_border += step
            jobs.append(j)
            j.start()

        for job in jobs:
            job.join()

        map_stop = timeit.default_timer()
        print("Map time for one frequent", map_stop - map_start)
        return map_result

    processes_size = 2
    start = timeit.default_timer()

    map_result = find_frequent_map(processes_size, dataset)
    print("Map for one frequent function finished")

    shuffle_result = shuffle_function(map_result)
    print("Shuffle for one frequent function finished")

    reduce_result = reduce_function(processes_size, shuffle_result, support_cnt)
    print("Reduce for one frequent function finished")

    result = convert_reduce_result(reduce_result)

    stop = timeit.default_timer()
    print("MapReduce for one frequent itemsets finished", stop - start)
    return result


def find_frequent_k(subset, trie, support_cnt):
    def map(trie, subsets, map_result, left_border, right_border):
        def count_support(node, target, iterator):
            if node.items == target:
                return 1
            current_item = next(iterator)
            node = binary_search(node.children, current_item)
            if node:
                return count_support(node, target, iterator)
            return 0

        result = {}
        for i in range(left_border, right_border):
            subset = subsets[i]
            if count_support(trie, subset, iter(subset)) == 1:
                subset = tuple(subset)
                if subset in result.keys():
                    result[subset] += 1
                else:
                    result[subset] = 1
        print(result)
        map_result.put(result)

    def find_frequent_map(processes_size, trie, subsets):
        map_start = timeit.default_timer()
        map_result = multiprocessing.Manager().Queue()
        jobs = []
        left_border = 0
        step = int(len(subsets) / processes_size)
        right_border = step
        for i in range(processes_size):
            j = Process(target=map,
                        args=(trie, subsets, map_result, left_border, right_border))
            print("run map with left " + str(left_border) + " and right " + str(right_border))
            left_border += step
            if j == processes_size - 1:
                right_border = len(subsets)
            else:
                right_border += step
            jobs.append(j)
            j.start()

        for job in jobs:
            job.join()

        map_stop = timeit.default_timer()
        print("map time = ", map_stop - map_start)
        return map_result

    processes_size = 2
    start = timeit.default_timer()

    map_result = find_frequent_map(processes_size, trie, subset)
    print("Map for k frequent function finished")

    shuffle_result = shuffle_function(map_result)
    print("Shuffle for k frequent function finished")
    print(shuffle_result)

    if shuffle_result:
        reduce_result = reduce_function(processes_size, shuffle_result, support_cnt)
        print("Reduce for k frequent function finished")
        result = convert_reduce_result(reduce_result)
    else:
        result = []

    stop = timeit.default_timer()
    print("MapReduce for k frequent itemsets finished", stop - start)
    return result
