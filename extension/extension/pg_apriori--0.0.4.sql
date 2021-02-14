\echo Use "CREATE EXTENSION pg_apriori" to load this file. \quit

CREATE TYPE apriori_type AS (support_table VARCHAR, rules_table VARCHAR);
CREATE OR REPLACE FUNCTION apriori(IN json_data VARCHAR) RETURNS SETOF apriori_type AS
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

    import csv, itertools, parameters


    def find_frequent_one(data_set, support):
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
            if cnt >= (support * total / 100):
                frequent_1.append(([key], cnt))
        return frequent_1


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

                        hash_key = self.hash(old_itemset[index])
                        if hash_key not in node.children:
                            node.children[hash_key] = HNode()
                        self.recur_insert(node.children[hash_key], old_itemset, index + 1, old_cnt)
                    del node.bucket
                    node.isLeaf = False
            else:
                hash_key = self.hash(itemset[index])
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
                hash_key = self.hash(itemset[index])
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

        def hash(self, val):
            return val % self.max_child_cnt


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
        for itemset in f_itemsets:
            hash_map[tuple(itemset[0])] = itemset[1]

        a_rules = []
        for itemset in f_itemsets:
            length = len(itemset[0])
            if length == 1:
                continue

            union_support = hash_map[tuple(itemset[0])]
            for i in range(1, length):

                lefts = map(list, itertools.combinations(itemset[0], i))
                for left in lefts:
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
            plpy.execute(insert_table_query % (item, support))
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
                             " VALUES (ARRAY[%s], ARRAY[%s], %1.3f)"

        plpy.execute(create_table_query)

        for rule, confidence in sorted(result_data, key=lambda x: x[1]):
            pre, post = rule
            if len(pre) > 1:
                pre = str(pre)[1:-1]
            else:
                pre = str(pre)[1:-2]
            if len(post) > 1:
                post = str(post)[1:-1]
            else:
                post = str(post)[1:-2]
            plpy.execute(insert_table_query % (pre, post, confidence))
        return result_table_name

    def prepare_result(support_result, rules):
        support_table_name = create_tmp_support_table(support_result)
        rules_table_name = create_tmp_rule_table(rules)
        return support_table_name, rules_table_name

    data = prepare_data_from_json(json_data)
    transactions = {}
    for row in plpy.cursor("select * from " + data.tale_name):
        item_column = data.item_column
        transaction_column = data.transaction_column
        if not row[transaction_column] in transactions:
            new_list = []
            new_list.append(row[item_column])
            transactions[row[transaction_column]] = new_list
        else:
            transactions[row[transaction_column]].append(row[item_column])
    frequent = apriori_generate_frequent_itemsets(data, parameters.SUPPORT)
    a_rules = generate_association_rules(frequent, parameters.CONFIDENCE)
    return [prepare_result(items, rules)]

$$
LANGUAGE 'plpython3u' VOLATILE;
