\echo Use "CREATE EXTENSION pg_apriori_parallel" to load this file. \quit

CREATE TYPE apriori_parallel_type AS (support_table VARCHAR, rules_table VARCHAR);
CREATE OR REPLACE FUNCTION apriori_parallel(IN json_data VARCHAR) RETURNS SETOF apriori_parallel_type AS
$$
    import json
    from collections import defaultdict
    from itertools import chain, combinations

    class Data:
        def __init__(self, table_name, transaction_column, item_column):
            self.tale_name = table_name
            self.transaction_column = transaction_column
            self.item_column = item_column


    json_attr = {"table_name", "transaction_column", "item_column"}


    def prepare_data_from_json(json_data):
        json_data = json.loads(json_data)
        keys_list = set()
        for key in json_data.keys():
            keys_list.add(key)
        if json_attr != keys_list:
            raise ValueError("Bad json")
        return Data(json_data["table_name"], json_data["transaction_column"], json_data["item_column"])

    import csv, itertools
    import multiprocessing
    from multiprocessing import Process

    def find_frequent_one(dataset, support_cnt):

        def separate_data_for_processes(processes_size, dataset):
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
            map_result = multiprocessing.Manager().Queue()
            jobs = []
            for i in range(processes_size):
                j = Process(target=find_frequent_one_map,
                            args=(separated_dataset[i], map_result))
                jobs.append(j)
                j.start()

            for job in jobs:
                job.join()

            return map_result

        def find_frequent_one_shuffle(map_result):
            shuffle_result = dict()
            while not map_result.empty():
                current = map_result.get()
                for key, value in current.items():
                    if key in shuffle_result:
                        shuffle_result[key].append(value)
                    else:
                        shuffle_result[key] = []
                        shuffle_result[key].append(value)
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

            return reduce_result

        processes_size = 2
        separated_dataset = separate_data_for_processes(processes_size, dataset)
        map_result = run_find_frequent_map(processes_size, separated_dataset)
        shuffle_result = find_frequent_one_shuffle(map_result)
        support_cnt = support_cnt * len(dataset) / 100
        reduce_result = run_find_frequent_reduce(processes_size, shuffle_result, support_cnt)
        result = []
        while not reduce_result.empty():
            current = reduce_result.get()
            for item, val in current.items():
                result.append(([item], val))
        return result


    class HNode:
        def __init__(self):
            self.children = {}
            self.isLeaf = True
            self.bucket = {}


    class HTree:
        def __init__(self, max_leaf_cnt, max_child_cnt):
            self.root = HNode()
            self.max_leaf_cnt = max_leaf_cnt
            self.max_child_cnt = max_child_cnt
            self.frequent_itemsets = []

        def recur_insert(self, node, itemset, index, cnt):
            if index == len(itemset):
                if itemset in node.bucket:
                    node.bucket[itemset] += cnt
                else:
                    node.bucket[itemset] = cnt
                return

            if node.isLeaf:

                if itemset in node.bucket:
                    node.bucket[itemset] += cnt
                else:
                    node.bucket[itemset] = cnt
                if len(node.bucket) == self.max_leaf_cnt:
                    for old_itemset, old_cnt in node.bucket.items():

                        hash_key = self.hash_function(old_itemset[index])
                        if hash_key not in node.children:
                            node.children[hash_key] = HNode()
                        self.recur_insert(node.children[hash_key], old_itemset, index + 1, old_cnt)
                    del node.bucket
                    node.isLeaf = False
            else:
                hash_key = self.hash_function(itemset[index])
                if hash_key not in node.children:
                    node.children[hash_key] = HNode()
                self.recur_insert(node.children[hash_key], itemset, index + 1, cnt)

        def insert(self, itemset):
            itemset = tuple(itemset)
            self.recur_insert(self.root, itemset, 0, 0)

        def add_support(self, itemset):
            runner = self.root
            itemset = tuple(itemset)
            index = 0
            while True:
                if runner.isLeaf:
                    if itemset in runner.bucket:
                        runner.bucket[itemset] += 1
                    break
                hash_key = self.hash_function(itemset[index])
                if hash_key in runner.children:
                    runner = runner.children[hash_key]
                else:
                    break
                index += 1

        def dfs(self, node, support_cnt):
            if node.isLeaf:
                for key, value in node.bucket.items():
                    if value >= support_cnt:
                        self.frequent_itemsets.append((list(key), value))
                return

            for child in node.children.values():
                self.dfs(child, support_cnt)

        def get_frequent_itemsets(self, support_cnt):
            self.frequent_itemsets = []
            self.dfs(self.root, support_cnt)
            return self.frequent_itemsets

        def hash_function(self, val):
            return hash(val) % self.max_child_cnt


    def generate_hash_tree(candidate_itemsets, length, max_leaf_cnt=4, max_child_cnt=5):
        htree = HTree(max_child_cnt, max_leaf_cnt)
        for itemset in candidate_itemsets:
            htree.insert(itemset)
        return htree


    def generate_k_subsets(dataset, length):
        subsets = []
        for row in dataset.values():
            subsets.extend(map(list, itertools.combinations(row, length)))
        return subsets


    def is_prefix(list_1, list_2):
        if len(list_1) == 1:
            return True
        return list_1[:len(list_1) - 1] == list_2[:len(list_2) - 1]


    def apriori_generate_frequent_itemsets(dataset, support):
        support_cnt = int(support / 100.0 * len(dataset))

        all_frequent_itemsets = find_frequent_one(dataset, support)

        prev_frequent = [x[0] for x in all_frequent_itemsets]
        length = 2

        while len(prev_frequent) > 1:
            new_candidates = []
            for i in range(len(prev_frequent)):
                j = i + 1
                while j < len(prev_frequent) and is_prefix(prev_frequent[i], prev_frequent[j]):
                    # this part makes sure that all of the items remain lexicographically sorted.
                    new_candidates.append(prev_frequent[i][:-1] +
                                          [prev_frequent[i][-1]] +
                                          [prev_frequent[j][-1]]
                                          )
                    j += 1

            h_tree = generate_hash_tree(new_candidates, length)
            k_subsets = generate_k_subsets(dataset, length)

            for subset in k_subsets:
                h_tree.add_support(subset)

            new_frequent = h_tree.get_frequent_itemsets(support_cnt)
            all_frequent_itemsets.extend(new_frequent)
            prev_frequent = [tup[0] for tup in new_frequent]
            prev_frequent.sort()
            length += 1

        return all_frequent_itemsets

    def generate_association_rules(f_itemsets, confidence):
        hash_map = {}
        sorted_itemsets = []
        for itemset in f_itemsets:
            arr = sorted(itemset[0])
            sorted_itemsets.append((arr, itemset[1]))
        for itemset in sorted_itemsets:
            hash_map[tuple(itemset[0])] = itemset[1]

        a_rules = []
        for itemset in sorted_itemsets:
            length = len(itemset[0])
            if length == 1:
                continue

            union_support = hash_map[tuple(itemset[0])]
            for i in range(1, length):

                lefts = map(list, itertools.combinations(itemset[0], i))
                for left in lefts:
                    if not tuple(left) in hash_map:
                        continue
                    conf = 100.0 * union_support / hash_map[tuple(left)]
                    if conf >= confidence:
                        a_rules.append([left, list(set(itemset[0]) - set(left)), conf])
        return a_rules


    from datetime import datetime

    def create_tmp_support_table(result_data):
        dt_string = datetime.now().strftime("%Y%m%d%H%M%S")
        result_table_name = "pg_apriori_support_" + dt_string
        create_table_query = "CREATE TABLE " + result_table_name +\
                             "(" + \
                             "items VARCHAR []," + \
                             "support double precision" +\
                             ")"

        insert_table_query = "INSERT INTO " + result_table_name +\
                             "(items, support)" + \
                             " VALUES (ARRAY%s, %1.3f)"

        plpy.execute(create_table_query)

        for item, support in result_data:
            item_string = list(map(lambda r: str(r), item))
            plpy.execute(insert_table_query % (item_string, support))
        return result_table_name


    def create_tmp_rule_table(result_data):
        dt_string = datetime.now().strftime("%Y%m%d%H%M%S")
        result_table_name = "pg_apriori_rules_" + dt_string
        create_table_query = "CREATE TABLE " + result_table_name +\
                             "(" + \
                             "items_from VARCHAR []," + \
                             "items_to VARCHAR []," + \
                             "confidence double precision" +\
                             ")"

        insert_table_query = "INSERT INTO " + result_table_name +\
                             "(items_from, items_to, confidence)" + \
                             " VALUES (ARRAY%s, ARRAY%s, %1.3f)"

        plpy.execute(create_table_query)

        for rule_from, rule_to, confidence in result_data:
            rule_from_string = list(map(lambda r: str(r), rule_from))
            rule_to_string = list(map(lambda r: str(r), rule_to))
            plpy.execute(insert_table_query % (rule_from_string, rule_to_string, confidence))

        return result_table_name

    def prepare_result(support_result, rules):
        support_table_name = create_tmp_support_table(support_result)
        rules_table_name = create_tmp_rule_table(rules)
        return support_table_name, rules_table_name

    dataset = prepare_data_from_json(json_data)
    transactions = {}
    for row in plpy.cursor("select * from " + dataset.tale_name):
        item_column = dataset.item_column
        transaction_column = dataset.transaction_column
        if not row[transaction_column] in transactions:
            new_list = []
            new_list.append(row[item_column])
            transactions[row[transaction_column]] = new_list
        else:
            transactions[row[transaction_column]].append(row[item_column])
    frequent = apriori_generate_frequent_itemsets(transactions, 3)
    a_rules = generate_association_rules(frequent, 5)
    return [prepare_result(frequent, a_rules)]

$$
LANGUAGE 'plpython3u' VOLATILE;
