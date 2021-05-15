import itertools
import multiprocessing
import timeit
from multiprocessing import Process


class TrieNode(object):
    def __init__(self, item, depth, items):
        self.item = item
        self.depth = depth
        self.items = items
        self.support = 0
        self.children = []
        self.invalid = False
        self.word_finished = False


def binary_search(array, target):
    lower = 0
    upper = len(array)
    if upper == lower:
        return None
    while lower < upper:
        x = lower + (upper - lower) // 2
        val = array[x].item
        if target == val:
            return array[x]
        elif target > val:
            if lower == x:
                break
            lower = x
        elif target < val:
            upper = x
    for i in (lower, upper - 1):
        if array[i].item == target:
            return array[i]
    return None


def count_support(node, target, iterator):
    if node.items == target:
        node.support += 1
        return
    current_item = next(iterator)
    node = binary_search(node.children, current_item)
    if node:
        count_support(node, target, iterator)


def add(root, items):
    print(items)
    current_node = root
    for item in items:
        print(item)
        found_node = binary_search(current_node.children, item)
        if found_node is not None:
            current_node = found_node
        else:
            new_node = TrieNode(item, current_node.depth + 1, current_node.items + [item])
            current_node.children.append(new_node)
            current_node = new_node
    # last
    current_node.word_finished = True


def search_candidates(visited, node, max_depth, used_candidates_items, candidate_items):
    node.support = 0
    if node.word_finished:
        node.word_finished = False
        if candidate_items:
            for item in candidate_items:
                if node.item != item:
                    add(node, [item])
        else:
            node.invalid = True
        return node.item
    if node.depth == max_depth - 1 or max_depth == 1:
        for i in range(len(node.children) - 1, -1, -1):
            neighbor = node.children[i]
            candidate = search_candidates(visited, neighbor, max_depth, used_candidates_items, candidate_items)
            if not candidate in used_candidates_items:
                used_candidates_items.add(candidate)
                candidate_items.insert(0, candidate)
            if neighbor.invalid:
                node.children.remove(neighbor)
    else:
        if node not in visited:
            visited.add(node)
            nodes_for_remove = []
            for i in range(len(node.children)):
                neighbor = node.children[i]
                search_candidates(visited, neighbor, max_depth, set(), list())
                if neighbor.invalid:
                    nodes_for_remove.append(neighbor)
            node.children = [x for x in node.children if (x not in nodes_for_remove)]
    if len(node.children) == 0:
        node.invalid = True


def dfs(visited, node):
    if node not in visited:
        if node.word_finished:
            print(node.items, node.support)
        visited.add(node)
        for neighbor in node.children:
            dfs(visited, neighbor)


def separate_data_for_processes(processes_size, dataset):
    separate_start = timeit.default_timer()
    datasets = []
    step = len(dataset) // processes_size
    border_last_full = step * (processes_size - 1)
    current_data = {}
    counter = 0
    for i in dataset.items():
        if 0 < counter <= border_last_full and counter % step == 0:
            datasets.append(current_data)
            current_data = {}
        current_data[i[0]] = i[1]
        counter += 1
    datasets.append(current_data)
    separate_stop = timeit.default_timer()

    if len(datasets) < processes_size:
        current_length = len(datasets)
        for i in range(processes_size - current_length):
            datasets.append({})
    print("Separate dataset time ", separate_stop - separate_start)

    return datasets


def convert_reduce_result(reduce_result, transactions_num):
    result = []
    while not reduce_result.empty():
        current = reduce_result.get()
        for item, val in current.items():
            result.append(([item], val / transactions_num))
    return result


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
    print(min_support)

    def reduce(map_result, min_support, reduce_result):
        result = dict()
        for key, value in map_result.items():
            current_count = 0
            for term in value:
                current_count += term
            print(current_count, " REDUCE ", key)

            if current_count >= min_support:
                print("will be result", key)
                result[key] = current_count
        reduce_result.put(result)

    reduce_start = timeit.default_timer()
    separated_dataset = separate_data_for_processes(processes_size, shuffle_result)

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


def find_frequent_one(dataset, support_cnt):
    def map(dataset, map_result, left, right):
        result = {}
        for i in range(left, right):
            for item in dataset[i]:
                if item in result:
                    result[item] += 1
                else:
                    result[item] = 1

        print("MAP RES: ", result)
        map_result.put(result)

    def find_frequent_map(processes_size, dataset):
        map_start = timeit.default_timer()
        map_result = multiprocessing.Manager().Queue()
        jobs = []
        left_border = 0
        step = len(dataset) // processes_size
        right_border = step
        for i in range(processes_size):
            if i == processes_size - 1:
                right_border = len(dataset) - left_border + 1
            j = Process(target=map,
                        args=(dataset, map_result, left_border, right_border))
            print("start map with left = %s and right = %s" % (left_border, right_border))
            left_border = right_border
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
    print(shuffle_result)

    reduce_result = reduce_function(processes_size, shuffle_result, support_cnt)
    print("Reduce for one frequent function finished")

    # result = convert_reduce_result(reduce_result, len(dataset))

    stop = timeit.default_timer()
    result = convert_reduce_result(reduce_result, len(dataset))

    print("MapReduce for one frequent itemsets finished", stop - start)
    return result


def find_frequent_k(subset, trie, support_cnt, transactions_num):
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
            if i == processes_size - 2:
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
        result = convert_reduce_result(reduce_result, transactions_num)
    else:
        result = []

    stop = timeit.default_timer()
    print(result)
    print("MapReduce for k frequent itemsets finished", stop - start)
    return result


def generate_k_subsets(dataset, length):
    subsets = []
    for row in dataset:
        subsets.extend(map(list, sorted(itertools.combinations(row, length))))
    return subsets


def generate_association_rules(f_itemsets, confidence):
    hash_map = {}
    for itemset in f_itemsets:
        value = itemset[1]
        if isinstance(itemset[0][0], tuple):
            itemset = itemset[0][0]
        else:
            itemset = tuple(itemset[0])
        hash_map[itemset] = value

    a_rules = []
    for itemset in f_itemsets:
        if isinstance(itemset[0][0], tuple):
            itemset = itemset[0][0]
        else:
            itemset = itemset[0]
        length = len(itemset)
        if length == 1:
            continue

        union_support = hash_map[itemset]
        for i in range(1, length):
            lefts = map(list, itertools.combinations(itemset, i))
            for left in lefts:
                if not tuple(left) in hash_map:
                    continue
                conf = 100.0 * union_support / hash_map[tuple(left)]
                if conf >= confidence:
                    a_rules.append([left, list(set(itemset) - set(left)), conf])
    return a_rules


def run(dataset, support_in_percent, confidence_in_percent):
    support = (support_in_percent * len(dataset) / 100)

    for i in range(len(dataset)):
        dataset[i] = sorted(dataset[i])

    frequent_one = find_frequent_one(dataset, support)
    frequent_one = sorted(frequent_one, key=lambda tup: tup[0])
    frequent_itemsets = frequent_one

    print("Founded frequent items with length 1")
    print(frequent_one)
    print("--------------------")
    current_candidates_tree = TrieNode(None, 0, [])
    for candidate in frequent_one:
        add(current_candidates_tree, [candidate])

    print("trie:")
    dfs(set(), current_candidates_tree)
    k = 2
    while current_candidates_tree.children and k <= len(frequent_one):
        search_candidates(set(), current_candidates_tree, k - 1, set(), list())

        dfs(set(), current_candidates_tree)
        print("Candidates generated")

        k_subsets = generate_k_subsets(dataset, k)
        print("Subsets generated")

        frequent_itemsets_k = find_frequent_k(k_subsets, current_candidates_tree, support, len(dataset))
        print("Frequent items with length %s generated" % k)

        frequent_itemsets_k = sorted(frequent_itemsets_k, key=lambda tup: tup[0])
        frequent_itemsets.extend(frequent_itemsets_k)
        print(frequent_itemsets_k)

        # build trie with new frequent itemsets for new generation
        current_candidates_tree = TrieNode(None, 0, [])
        for candidate in frequent_itemsets_k:
            add(current_candidates_tree, list(candidate[0][0]))
        # dfs(set(), current_candidates_tree)
        print("New trie generated")

        k += 1

    a_rules = generate_association_rules(frequent_itemsets, confidence_in_percent)
    print(frequent_itemsets)
    print(len(frequent_itemsets))
    print('rules')
    print(a_rules)
    return frequent_itemsets, a_rules


# a = [(['ASIAN'], 0.2692307692307692), (['BLACK'], 0.3269230769230769), (['Bronx'], 0.11538461538461539), (['HISPANIC'], 0.15384615384615385), (['LBE'], 0.1346153846153846), (['MBE'], 0.75), (['NON-MINORITY'], 0.17307692307692307), (['New York'], 0.3269230769230769), (['WBE'], 0.38461538461538464), ([('ASIAN', 'MBE')], 0.2692307692307692), ([('ASIAN', 'New York')], 0.11538461538461539), ([('BLACK', 'MBE')], 0.3269230769230769), ([('Bronx', 'MBE')], 0.11538461538461539), ([('HISPANIC', 'MBE')], 0.15384615384615385), ([('MBE', 'New York')], 0.25), ([('MBE', 'WBE')], 0.19230769230769232), ([('NON-MINORITY', 'WBE')], 0.17307692307692307), ([('New York', 'WBE')], 0.17307692307692307), ([('ASIAN', 'MBE', 'New York')], 0.11538461538461539), ([('MBE', 'New York', 'WBE')], 0.11538461538461539)]
# generate_association_rules(a, parameters.CONFIDENCE)

transactions = [['eggs', 'bacon', 'soup'],
                ['eggs', 'bacon', 'apple'],
                ['soup', 'bacon', 'banana']]
frequent_items, rules = run(transactions, 0, 0)
print(frequent_items)
print(rules)

# Separate dataset time  5.142999725649133e-06
# Map time for one frequent 0.013218720999248035
# Map for one frequent function finished
# Shuffle time 0.0007920149992060033
# Shuffle for one frequent function finished
# {'bacon': [1, 2], 'eggs': [1, 1], 'soup': [1, 1], 'apple': [1], 'banana': [1]}
# Separate dataset time  3.967999873566441e-06
# [{'bacon': [1, 2], 'eggs': [1, 1]}, {'soup': [1, 1], 'apple': [1], 'banana': [1]}]
# {'bacon': [1, 2], 'eggs': [1, 1], 'soup': [1, 1], 'apple': [1], 'banana': [1]}
# 3  REDUCE  bacon
# 2  REDUCE  eggs
# 2  REDUCE  soup
# 1  REDUCE  apple
# 1  REDUCE  banana
# reduce time  0.010320745000171883
# Reduce for one frequent function finished
# MapReduce for one frequent itemsets finished 0.025179706000017177
# Founded frequent items with length 1
# [(['apple'], 0.3333333333333333), (['bacon'], 1.0), (['banana'], 0.3333333333333333), (['eggs'], 0.6666666666666666), (['soup'], 0.6666666666666666)], 0.3333333333333333), (['bacon'], 1.0), (['banana'], 0.3333333333333333), (['eggs'], 0.6666666666666666), (['soup'], 0.6666666666666666)]
