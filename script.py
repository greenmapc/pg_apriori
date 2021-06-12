import json
import itertools
import multiprocessing
import timeit
from multiprocessing import Process


class Data:
    def __init__(self, table_name, transaction_column, item_column, min_support=50, min_confidence=50):
        self.tale_name = table_name
        self.transaction_column = transaction_column
        self.item_column = item_column
        self.min_support = int(min_support)
        self.min_confidence = int(min_confidence)


json_attr = {"table_name", "transaction_column", "item_column", "min_support", "min_confidence"}


def prepare_data_from_json(json_data):
    json_data = json.loads(json_data)
    keys_list = set()
    for key in json_data.keys():
        keys_list.add(key)
    if json_attr != keys_list:
        raise ValueError("Bad json")
    return Data(json_data["table_name"], json_data["transaction_column"], json_data["item_column"],
                json_data["min_support"], json_data["min_confidence"])


class TrieNode(object):
    def __init__(self, item, depth, items):
        self.item = item
        self.depth = depth
        self.items = items
        self.children = []
        self.is_list = False


def list_binary_search(list, target):
    left = 0
    right = len(list) - 1
    while left < right - 1:
        middle = (left + right) // 2
        if list[middle] > target:
            right = middle
        else:
            left = middle
    for i in (left, right):
        if list[i] == target:
            return list[i]
    return None

def trie_binary_search(array, target):
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
    if array[lower].item == target:
        return array[lower]
    return None

def add(root, items):
    current_node = root
    for item in items:
        found_node = trie_binary_search(current_node.children, item)
        if found_node is not None:
            current_node = found_node
        else:
            new_node = TrieNode(item, current_node.depth + 1, current_node.items + [item])
            current_node.children.append(new_node)
            current_node = new_node
    # last
    current_node.is_list = True

def search_candidates(visited, node, new_trie_node, max_depth, size=0, edges=0):
    if node in visited:
        return 0

    invalid_depth = max_depth < 1

    if invalid_depth or node.depth == max_depth:
        children = node.children
        j = 1
        border = 0
        if invalid_depth:
            border = 1
        for i in range(len(children) - border):
            child = children[i]
            new_parent = TrieNode(child.item, new_trie_node.depth + 1, new_trie_node.items + [child.item])
            new_trie_node.children.append(new_parent)
            if j < len(children):
                edges += 1
            for new_child in children[j:]:
                new_node = TrieNode(new_child.item, new_parent.depth + 1, new_parent.items + [new_child.item])
                new_node.is_list = True
                new_parent.children.append(new_node)
                size += 1
                edges += 1
            j += 1
        return size, edges

    visited.add(node)
    for child in node.children:
        new_child = TrieNode(child.item, new_trie_node.depth + 1, new_trie_node.items + [child.item])
        new_trie_node.children.append(new_child)
        cur_size, cur_edges = search_candidates(visited, child, new_child, max_depth, 0)
        size += cur_size
        if cur_edges > 0:
            edges += 1 + cur_edges

    return size, edges

def separate_data_for_processes(processes_size, dataset):
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

    if len(datasets) < processes_size:
        current_length = len(datasets)
        for i in range(processes_size - current_length):
            datasets.append({})

    return datasets


def shuffle_function(map_result):
    shuffle_start = timeit.default_timer()
    shuffle_result = dict()
    while not map_result.empty():
        current = map_result.get()
        for key, value in current.items():
            if key in shuffle_result:
                shuffle_result[key] += value
            else:
                shuffle_result[key] = 0
                shuffle_result[key] += value
    shuffle_stop = timeit.default_timer()
    print('Shuffle time', shuffle_stop - shuffle_start)
    return shuffle_result


def reduce_function(processes_size, shuffle_result, min_support, transactions_num):
    def reduce(map_result, min_support, reduce_result, transactions_num):
        result = dict()
        reduce_process_start = timeit.default_timer()
        for key, value in map_result.items():
            if value >= min_support:
                result[key] = value / transactions_num
        reduce_result.update(result)
        reduce_process_finish = timeit.default_timer()
        print("One process reduce time ", reduce_process_finish - reduce_process_start)

    reduce_start = timeit.default_timer()
    separated_dataset = separate_data_for_processes(processes_size, shuffle_result)

    reduce_result = multiprocessing.Manager().dict()
    jobs = []
    for i in range(processes_size):
        j = Process(target=reduce,
                    args=(separated_dataset[i], min_support, reduce_result, transactions_num))
        jobs.append(j)
        j.start()

    for job in jobs:
        job.join()

    reduce_stop = timeit.default_timer()
    print("reduce time ", reduce_stop - reduce_start)
    return reduce_result

def find_frequent_one(dataset, support_cnt, processes_size):
    def map(dataset, map_result, left, right):
        result = {}
        for i in range(left, right):
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
        step = len(dataset) // processes_size
        right_border = step
        for i in range(processes_size):
            if i == processes_size - 1:
                right_border = len(dataset)

            j = Process(target=map,
                        args=(dataset, map_result, left_border, right_border))
            left_border = right_border
            right_border += step
            jobs.append(j)
            j.start()

        for job in jobs:
            job.join()

        map_stop = timeit.default_timer()
        print("Map time for one frequent", map_stop - map_start)
        return map_result

    start = timeit.default_timer()

    map_result = find_frequent_map(processes_size, dataset)

    shuffle_result = shuffle_function(map_result)

    reduce_result = reduce_function(processes_size, shuffle_result, support_cnt, len(dataset))

    stop = timeit.default_timer()
    print("MapReduce for one frequent itemsets finished", stop - start)
    return reduce_result

def find_frequent_k(transactions, trie, support_cnt, transactions_num, edges, k, processes_size):
    def map(transactions, map_result, t_left_border, t_right_border):

        def support_counter_with_iter_by_candidates(transaction, node, result):
            for child in node.children:
                if list_binary_search(transaction, child.item):
                    if child.is_list:
                        subset = tuple(child.items)
                        if subset in result.keys():
                            result[subset] += 1
                        else:
                            result[subset] = 1
                    else:
                        support_counter_with_iter_by_candidates(transaction, child, result)

        result = {}

        for i in range(t_left_border, t_right_border):
            transaction = transactions[i]
            if len(transaction) < k:
                continue
            support_counter_with_iter_by_candidates(transaction, trie, result)

        map_result.put(result)

    def find_frequent_map(processes_size, transactions):
        map_start = timeit.default_timer()
        map_result = multiprocessing.Manager().Queue()
        jobs = []
        t_left_border = 0
        t_step = int(len(transactions) / processes_size)
        t_right_border = t_step
        for i in range(processes_size):
            j = Process(target=map,
                        args=(transactions, map_result, t_left_border, t_right_border))
            t_left_border += t_step
            if i == processes_size - 2:
                t_right_border = len(transactions)
            else:
                t_right_border += t_step
            jobs.append(j)
            j.start()

        for job in jobs:
            job.join()

        map_stop = timeit.default_timer()
        print("Map step for find freq_k", map_stop - map_start)
        return map_result

    map_result = find_frequent_map(processes_size, transactions)

    shuffle_result = shuffle_function(map_result)
    if shuffle_result:
        reduce_result = reduce_function(processes_size, shuffle_result, support_cnt, transactions_num)
        result = reduce_result
    else:
        result = {}

    return result

def generate_association_rules(f_itemsets, confidence):
    hash_map = {}
    for itemset in f_itemsets:
        value = itemset[1]
        if isinstance(itemset[0], tuple):
            itemset = itemset[0]
        else:
            itemset = tuple([itemset[0]])
        hash_map[itemset] = value

    a_rules = []

    for itemset in f_itemsets:

        if isinstance(itemset[0], tuple):
            itemset = itemset[0]
        else:
            itemset = tuple([itemset[0]])

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
    processes_size = multiprocessing.cpu_count()

    for i in range(len(dataset)):
        dataset[i] = sorted(dataset[i])

    frequent_one = list(find_frequent_one(dataset, support, processes_size).items())

    frequent_one = sorted(frequent_one, key=lambda tup: tup[0])
    frequent_itemsets = frequent_one

    current_candidates_tree = TrieNode(None, 0, [])
    for candidate in frequent_one:
        add(current_candidates_tree, [candidate[0]])
    print("Trie from 1 nodes build")
    print("Founded frequent items with length 1")

    k = 2
    while current_candidates_tree.children and k <= len(frequent_one):
        start_c = timeit.default_timer()
        k_candidates_trie = TrieNode(None, 0, [])
        candidates_size, edges = search_candidates(set(), current_candidates_tree, k_candidates_trie, k - 2)

        print("Found %s candidates" % candidates_size)
        print("Found %s edges" % edges)

        finish_c = timeit.default_timer()

        print("Candidates generated for k = %s:" % k, finish_c - start_c)

        if candidates_size == 0:
            break

        start_freq_k = timeit.default_timer()

        frequent_itemsets_k = find_frequent_k(dataset, k_candidates_trie, support, len(dataset), edges, k,
                                              processes_size)

        finish_found = timeit.default_timer()

        print("Find freq k = %s:" % k, finish_found - start_freq_k)
        print("Frequent items with length %s generated" % k)

        data_preparing_start = timeit.default_timer()
        frequent_itemsets_k = list(frequent_itemsets_k.items())
        frequent_itemsets_k = sorted(frequent_itemsets_k, key=lambda tup: tup[0])

        frequent_itemsets.extend(frequent_itemsets_k)

        # build trie with new frequent itemsets for new generation
        current_candidates_tree = TrieNode(None, 0, [])
        for candidate in frequent_itemsets_k:
            add(current_candidates_tree, candidate[0])

        finish_preparing = timeit.default_timer()

        print("Prepared data for k = %s:" % k, finish_preparing - data_preparing_start)

        k += 1

    print("Found frequent itemsets")
    a_rules = generate_association_rules(frequent_itemsets, confidence_in_percent)

    return frequent_itemsets, a_rules


from datetime import datetime


def create_tmp_support_table(result_data, transactions_num):
    dt_string = datetime.now().strftime("%Y%m%d%H%M%S")
    result_table_name = "pg_apriori_support_" + dt_string
    create_table_query = "CREATE TABLE " + result_table_name + \
                         "(" + \
                         "items VARCHAR []," + \
                         "support double precision" + \
                         ")"

    insert_table_query = "INSERT INTO " + result_table_name + \
                         "(items, support)" + \
                         " VALUES (ARRAY%s, %1.3f)"

    # plpy.execute(create_table_query)
    print(create_table_query)

    for item, support in result_data:
        if isinstance(item, tuple):
            item = list(item)
        else:
            item = [item]
        # plpy.execute(insert_table_query % (item_string, support))
        print(insert_table_query % (item, support * 100))
    return result_table_name


def create_tmp_rule_table(result_data):
    dt_string = datetime.now().strftime("%Y%m%d%H%M%S")
    result_table_name = "pg_apriori_rules_" + dt_string
    create_table_query = "CREATE TABLE " + result_table_name + \
                         "(" + \
                         "items_from VARCHAR []," + \
                         "items_to VARCHAR []," + \
                         "confidence double precision" + \
                         ")"

    insert_table_query = "INSERT INTO " + result_table_name + \
                         "(items_from, items_to, confidence)" + \
                         " VALUES (ARRAY%s, ARRAY%s, %1.3f)"

    print(create_table_query)
    # plpy.execute(create_table_query)
    print(create_table_query)

    for rule_from, rule_to, confidence in result_data:
        rule_from_string = list(map(lambda r: str(r), rule_from))
        rule_to_string = list(map(lambda r: str(r), rule_to))
        # plpy.execute(insert_table_query % (rule_from_string, rule_to_string, confidence))
        print(insert_table_query % (rule_from_string, rule_to_string, confidence))

    return result_table_name


def prepare_result(support_result, rules, transactions_num):
    support_table_name = create_tmp_support_table(support_result, transactions_num)
    rules_table_name = create_tmp_rule_table(rules)
    return support_table_name, rules_table_name


import psycopg2


def run_with_postgres():
    con = psycopg2.connect(database="diploma", user="postgres", password="postgres", host="127.0.0.1", port="5432")
    print("Database opened successfully")

    cur = con.cursor()

    # print(cur.fetchall())

    json_data = '{ "table_name":"million_data_table", ' \
                '"transaction_column":"who", ' \
                '"item_column":"what",' \
                '"min_support": 3,' \
                '"min_confidence": 5}'
    user_data = prepare_data_from_json(json_data)
    transactions = {}
    cur.execute('''SELECT * FROM iter1_test_table''')
    for row in cur.fetchall():
        item_column = 1
        transaction_column = 0
        if not row[transaction_column] in transactions:
            new_list = []
            new_list.append(row[item_column])
            transactions[row[transaction_column]] = new_list
        else:
            transactions[row[transaction_column]].append(row[item_column])

    con.commit()
    con.close()
    frequent, a_rules = run(list(transactions.values()), user_data.min_support, user_data.min_confidence)
    prepare_result(frequent, a_rules, len(transactions.keys()))


if __name__ == '__main__':
    run_with_postgres()
