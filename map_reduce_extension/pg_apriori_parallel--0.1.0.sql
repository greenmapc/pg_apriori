\echo Use "CREATE EXTENSION pg_apriori_parallel" to load this file. \quit

CREATE TYPE apriori_parallel_type AS (support_table VARCHAR, rules_table VARCHAR);
CREATE OR REPLACE FUNCTION apriori_parallel(IN json_data VARCHAR) RETURNS SETOF apriori_parallel_type AS
$$
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
        return Data(json_data["table_name"], json_data["transaction_column"], json_data["item_column"], json_data["min_support"], json_data["min_confidence"])


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


    def find_frequent_itemsets(node, support):
        if node.word_finished:
            if node.support < support:
                node.invalid = True
                return []
            else:
                return [(node.items, node.support)]
        else:
            result = []
            for i in range(len(node.children) - 1, -1, -1):
                nodes = find_frequent_itemsets(node.children[i], support)
                result.extend(nodes)
                if node.children[i].invalid:
                    node.children.remove(node.children[i])
            if len(node.children) == 0:
                node.invalid = True
            return result


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


    def count_support(node, target, iterator):
        if node.items == target:
            node.support += 1
            return
        current_item = next(iterator)
        node = binary_search(node.children, current_item)
        if node:
            count_support(node, target, iterator)


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
        plpy.notice("Separate dataset time ", separate_stop - separate_start)

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
        plpy.notice('Shuffle time', shuffle_stop - shuffle_start)
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
        plpy.notice(separated_dataset)
        plpy.notice(shuffle_result)

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
        plpy.notice("reduce time ", reduce_stop - reduce_start)
        return reduce_result


    def convert_reduce_result(reduce_result):
        result = []
        while not reduce_result.empty():
            current = reduce_result.get()
            for item, val in current.items():
                result.append(([item], val))
        return result


    def find_frequent_one(dataset, support_cnt):
        def map(dataset, map_result):
            result = {}
            for key, values in dataset.items():
                for item in values:
                    if item in result:
                        result[item] += 1
                    else:
                        result[item] = 1
            map_result.put(result)

        def find_frequent_map(processes_size, dataset):
            map_start = timeit.default_timer()
            map_result = multiprocessing.Manager().Queue()
            jobs = []
            separated_dataset = separate_data_for_processes(processes_size, dataset)
            for i in range(processes_size):
                j = Process(target=map,
                            args=(separated_dataset[i], map_result))
                jobs.append(j)
                j.start()

            for job in jobs:
                job.join()

            map_stop = timeit.default_timer()
            plpy.notice("Map time for one frequent", map_stop - map_start)
            return map_result

        processes_size = 2
        start = timeit.default_timer()

        map_result = find_frequent_map(processes_size, dataset)
        plpy.notice("Map for one frequent function finished")

        shuffle_result = shuffle_function(map_result)
        plpy.notice("Shuffle for one frequent function finished")

        reduce_result = reduce_function(processes_size, shuffle_result, support_cnt)
        plpy.notice("Reduce for one frequent function finished")

        result = convert_reduce_result(reduce_result)

        stop = timeit.default_timer()
        plpy.notice("MapReduce for one frequent itemsets finished", stop - start)
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
                plpy.notice("run map with left " + str(left_border) + " and right " + str(right_border))
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
            plpy.notice("map time = ", map_stop - map_start)
            return map_result

        processes_size = 2
        start = timeit.default_timer()

        map_result = find_frequent_map(processes_size, trie, subset)
        plpy.notice("Map for k frequent function finished")

        shuffle_result = shuffle_function(map_result)
        plpy.notice("Shuffle for k frequent function finished")
        plpy.notice(shuffle_result)

        if shuffle_result:
            reduce_result = reduce_function(processes_size, shuffle_result, support_cnt)
            plpy.notice("Reduce for k frequent function finished")
            result = convert_reduce_result(reduce_result)
        else:
            result = []

        stop = timeit.default_timer()
        plpy.notice("MapReduce for k frequent itemsets finished", stop - start)
        return result


    def generate_k_subsets(dataset, length):
        subsets = []
        for row in dataset.values():
            subsets.extend(map(list, sorted(itertools.combinations(row, length))))
        return subsets


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


    def run(dataset, support_in_percent, confidence_in_percent):
        support = (support_in_percent * len(dataset) / 100)

        for key, transaction in dataset.items():
            dataset[key] = sorted(transaction)

        frequent_one = find_frequent_one(dataset, support)
        frequent_one = sorted(frequent_one, key=lambda tup: tup[0])
        frequent_itemsets = frequent_one

        plpy.notice("Founded frequent items with length 1")
        plpy.notice(frequent_one)

        current_candidates_tree = TrieNode(None, 0, [])
        for candidate in frequent_one:
            add(current_candidates_tree, candidate[0])

        k = 2
        while current_candidates_tree.children and k <= len(frequent_one):
            search_candidates(set(), current_candidates_tree, k - 1, set(), list())

            # dfs(set(), current_candidates_tree)
            plpy.notice("Candidates generated")

            k_subsets = generate_k_subsets(dataset, k)
            plpy.notice("Subsets generated")

            frequent_itemsets_k = find_frequent_k(k_subsets, current_candidates_tree, support)
            plpy.notice("Frequent items with length %s generated" % k)

            frequent_itemsets_k = sorted(frequent_itemsets_k, key=lambda tup: tup[0])
            frequent_itemsets.extend(frequent_itemsets_k)
            plpy.notice(frequent_itemsets_k)

            # build trie with new frequent itemsets for new generation
            current_candidates_tree = TrieNode(None, 0, [])
            for candidate in frequent_itemsets_k:
                add(current_candidates_tree, list(candidate[0][0]))
            # dfs(set(), current_candidates_tree)
            plpy.notice("New trie generated")

            k += 1

        a_rules = generate_association_rules(frequent_itemsets, confidence_in_percent)
        return frequent_itemsets, a_rules


    from datetime import datetime


    def create_tmp_support_table(result_data):
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

        plpy.execute(create_table_query)

        for item, support in result_data:
            if isinstance(item[0], tuple):
                item = list(item[0])
            item_string = list(map(lambda r: str(r), item))
            plpy.execute(insert_table_query % (item_string, support))
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


    user_data = prepare_data_from_json(json_data)
    transactions = {}
    for row in plpy.cursor("select * from " + user_data.tale_name):
        item_column = user_data.item_column
        transaction_column = user_data.transaction_column
        if not row[transaction_column] in transactions:
            new_list = []
            new_list.append(row[item_column])
            transactions[row[transaction_column]] = new_list
        else:
            transactions[row[transaction_column]].append(row[item_column])
    plpy.notice(transactions)
    frequent, a_rules = run(transactions, user_data.min_support, user_data.min_confidence)
    return [prepare_result(frequent, a_rules)]


$$
LANGUAGE 'plpython3u' VOLATILE;
