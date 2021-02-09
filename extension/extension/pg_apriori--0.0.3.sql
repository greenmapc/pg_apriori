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


    def subsets(arr):
        """ Returns non empty subsets of arr"""
        return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


    def return_items_with_min_support(item_set, transaction_list, min_support, freq_set):
        """calculates the support for items in the itemSet and returns a subset
        of the itemSet each of whose elements satisfies the minimum support"""
        item_set_with_min_support = set()
        local_set = defaultdict(int)

        # todo требует оптимизации
        # todo необходимо использовать хэш-дерево
        for item in item_set:
            for transaction in transaction_list:
                if item.issubset(transaction):
                    freq_set[item] += 1
                    local_set[item] += 1

        for item, count in local_set.items():
            support = float(count) / len(transaction_list)

            if support >= min_support:
                item_set_with_min_support.add(item)

        return item_set_with_min_support

        # генерация кандидатов
        # todo генерировать в отсортированном порядке для хэш дерева


    def generate_candidates(item_set, length):
        """Join a set with itself and returns the n-element itemsets"""
        return set(
            [i.union(j) for i in item_set for j in item_set if len(i.union(j)) == length]
        )

        # get list of transactions and set of all transactions elements


    def get_item_set_transaction_list(transtactions):
        transaction_list = list()
        item_set = set()
        for record in transtactions.values():
            transaction = frozenset(record)
            transaction_list.append(transaction)
            for item in transaction:
                item_set.add(frozenset([item]))
        return item_set, transaction_list


    def run_apriori(data_iter, min_support, min_confidence):
        """
        run the apriori algorithm. data_iter is a record iterator
        Return both:
         - items (tuple, support)
         - rules ((pretuple, posttuple), confidence)
        """
        item_set, transaction_list = get_item_set_transaction_list(data_iter)

        # map of item sets with frequency
        # (key=n-itemSets,value=support)
        freq_set = defaultdict(int)

        # key = item set length
        # value = list of item sets
        large_set = dict()

        # find more frequently item set
        one_item_satisfied_set = return_items_with_min_support(item_set, transaction_list, min_support, freq_set)

        current_large_set = one_item_satisfied_set

        item_length = 2
        while current_large_set != set([]):
            large_set[item_length - 1] = current_large_set
            current_large_set = generate_candidates(current_large_set, item_length)
            current_satisfied_set = return_items_with_min_support(
                current_large_set, transaction_list, min_support, freq_set
            )
            current_large_set = current_satisfied_set
            item_length = item_length + 1

        def get_support(item):
            """local function which Returns the support of an item"""
            return float(freq_set[item]) / len(transaction_list)

        # list of tuple (item set, support)
        items_result = []
        for key, value in large_set.items():
            # calculate support for each item set
            items_result.extend([(tuple(item), get_support(item)) for item in value])

        # list of tuple (item set, confidence)
        rules_result = []
        for key, value in list(large_set.items())[1:]:
            for item in value:
                # separate set by elements
                _subsets = map(frozenset, [x for x in subsets(item)])
                # calculate confidence for each element of item set
                for element in _subsets:
                    remain = item.difference(element)
                    if len(remain) > 0:
                        confidence = get_support(item) / get_support(element)
                        if confidence >= min_confidence:
                            rules_result.append(((tuple(element), tuple(remain)), confidence))
        return items_result, rules_result

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
                             " VALUES (ARRAY[%s], %1.3f)"

        plpy.execute(create_table_query)

        for item, support in sorted(result_data, key=lambda x: x[1]):
            if len(item) == 1:
                item_list = str(item)[1:-2]
            else:
                item_list = str(item)[1:-1]
            plpy.execute(insert_table_query % (item_list, support))
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
    items, rules = run_apriori(transactions, 0.17, 0.68)
    return [prepare_result(items, rules)]


$$
LANGUAGE 'plpython3u' VOLATILE;
