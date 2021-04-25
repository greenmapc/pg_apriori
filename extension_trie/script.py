import itertools
import json


class Data:
    def __init__(self, table_name, transaction_column, item_column, min_support=50, min_confidence=50):
        self.tale_name = table_name
        self.transaction_column = transaction_column
        self.item_column = item_column
        self.min_support = int(min_support)
        self.min_confidence = int(min_confidence)


json_attr = {"table_name", "transaction_column", "item_column", "min_support", "min_confidence"}


def dfs(visited, node):
    if node not in visited:
        if node.word_finished:
            print(node.items, node.support)
        visited.add(node)
        for neighbor in node.children:
            dfs(visited, neighbor)


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


def add(root, items):
    current_node = root
    for item in items:
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


def find_frequent_one(data_set, support):
    candidate_one = {}
    for row in data_set.values():
        for val in row:
            if val in candidate_one:
                candidate_one[val] += 1
            else:
                candidate_one[val] = 1
    frequent_1 = []
    for key, cnt in candidate_one.items():
        if cnt >= support:
            frequent_1.append(([key], cnt))
    return frequent_1


def find_frequent_k(subsets, trie, support_cnt):
    def calculate_node_support(node, target, iterator):
        if node.items == target:
            node.support += 1
            return
        current_item = next(iterator)
        node = binary_search(node.children, current_item)
        if node:
            calculate_node_support(node, target, iterator)

    def find_frequent_items(visited, node, min_support, result):
        if node not in visited:
            if node.word_finished:
                if node.support >= min_support:
                    result.append((node.items, node.support))
            visited.add(node)
            for neighbor in node.children:
                find_frequent_items(visited, neighbor, min_support, result)

    for subset in subsets:
        calculate_node_support(trie, subset, iter(subset))

    result = list()
    find_frequent_items(set(), trie, support_cnt, result)
    return result


def generate_k_subsets(dataset, length):
    subsets = []
    for row in dataset.values():
        subsets.extend(map(list, sorted(itertools.combinations(row, length))))
    return subsets


def generate_association_rules(f_itemsets, confidence):
    hash_map = {}
    for itemset in f_itemsets:
        value = itemset[1]
        itemset = tuple(itemset[0])
        hash_map[itemset] = value

    a_rules = []
    for itemset in f_itemsets:
        itemset = itemset[0]
        length = len(itemset)
        if length == 1:
            continue
        itemset = tuple(itemset)
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
    print(support)

    for key, transaction in dataset.items():
        dataset[key] = sorted(transaction)

    print("frequent one found start")
    frequent_one = find_frequent_one(dataset, support)
    frequent_one = sorted(frequent_one, key=lambda tup: tup[0])
    frequent_itemsets = frequent_one

    print("Founded frequent items with length 1")
    print(frequent_one)

    current_candidates_tree = TrieNode(None, 0, [])
    for candidate in frequent_one:
        add(current_candidates_tree, candidate[0])

    k = 2
    while current_candidates_tree.children and k <= len(frequent_one):
        search_candidates(set(), current_candidates_tree, k - 1, set(), list())

        dfs(set(), current_candidates_tree)
        print("Candidates generated")

        k_subsets = generate_k_subsets(dataset, k)
        print("Subsets generated")

        frequent_itemsets_k = find_frequent_k(k_subsets, current_candidates_tree, support)
        print("Frequent items with length %s generated" % k)

        frequent_itemsets_k = sorted(frequent_itemsets_k, key=lambda tup: tup[0])
        frequent_itemsets.extend(frequent_itemsets_k)
        print(frequent_itemsets_k)

        # build trie with new frequent itemsets for new generation
        current_candidates_tree = TrieNode(None, 0, [])
        for candidate in frequent_itemsets_k:
            add(current_candidates_tree, list(candidate[0]))
        dfs(set(), current_candidates_tree)
        print("New trie generated")

        k += 1

    print(len(frequent_itemsets))
    print(frequent_itemsets)
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
        if isinstance(item[0], tuple):
            item = list(item[0])
        item_string = list(map(lambda r: str(r), item))
        # plpy.execute(insert_table_query % (item_string, support))
        print(insert_table_query % (item_string, support / transactions_num * 100))
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
    print(transactions)
    print(len(transactions))
    frequent, a_rules = run(transactions, user_data.min_support, user_data.min_confidence)
    print(len(a_rules))
    print(a_rules)
    prepare_result(frequent, a_rules, len(transactions.keys()))


if __name__ == '__main__':
    run_with_postgres()
