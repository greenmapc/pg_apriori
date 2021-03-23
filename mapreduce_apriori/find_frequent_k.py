import multiprocessing
import timeit
from multiprocessing import Process

from mapreduce_apriori.trie import binary_search


def find_frequent_k(subset, trie, support_cnt):
    def separate_data_for_processes(processes_size, dataset):
        separate_start = timeit.default_timer()
        datasets = []
        step = int(len(dataset) / processes_size)
        border = step
        current_data = {}
        item_num = 0
        if len(datasets) <= border:
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
        separate_stop = timeit.default_timer()
        print("separate dataset time ", separate_stop - separate_start)
        return datasets

    def find_frequent_k_map(trie, subsets, map_result, left_border, right_border):
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

    def find_frequent_k_shuffle(map_result):
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

    def run_find_frequent_k_map(processes_size, trie, subsets):
        map_start = timeit.default_timer()
        map_result = multiprocessing.Manager().Queue()
        jobs = []
        left_border = 0
        step = int(len(subsets) / processes_size)
        right_border = step
        for i in range(processes_size):
            j = Process(target=find_frequent_k_map,
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

    def find_frequent_k_reduce(map_result, min_support, reduce_result):
        result = dict()
        for key, value in map_result.items():
            current_count = 0
            for term in value:
                current_count += term
            if current_count >= min_support:
                result[key] = current_count
        reduce_result.put(result)
        # reduce_result.append(result)

    def run_find_frequent_k_reduce(processes_size, separated_data, min_support):
        reduce_start = timeit.default_timer()
        reduce_result = multiprocessing.Manager().Queue()
        # reduce_result = []
        jobs = []
        # find_frequent_k_reduce(shuffle_result, find_frequent_k_reducemin_support, reduce_result, 0, len(shuffle_result.keys()))
        for i in range(processes_size):
            if len(shuffle_result) > i + 1:
                j = Process(target=find_frequent_k_reduce,
                            args=(separated_data[i], min_support, reduce_result))
                jobs.append(j)
                j.start()

        for job in jobs:
            job.join()

        reduce_stop = timeit.default_timer()
        print("reduce time ", reduce_stop - reduce_start)
        return reduce_result

    print(support_cnt)
    processes_size = 1
    start = timeit.default_timer()
    map_result = run_find_frequent_k_map(processes_size, trie, subset)
    print("map function finished")
    shuffle_result = find_frequent_k_shuffle(map_result)
    print("shuffle function finished")
    print(shuffle_result)
    separated_shuffle_data = separate_data_for_processes(processes_size, shuffle_result)
    print("separated shuffle result", separated_shuffle_data)
    reduce_result = run_find_frequent_k_reduce(processes_size, separated_shuffle_data, support_cnt)
    print("reduce function finished")
    # check_sum = 0
    result = []
    stop = timeit.default_timer()
    print("MAP REDUCE TIME", stop - start)
    while not reduce_result.empty():
        current = reduce_result.get()
        for item, val in current.items():
            result.append(([item], val))
        # print(current)
    # # print(check_sum)
    # print(len(result))
    # print(result)
    return result
