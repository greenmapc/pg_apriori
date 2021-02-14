import csv, itertools, parameters


def load_data(filename):
    """
    Loads transactions from given file
    :param filename:
    :return:
    """
    reader = csv.reader(open(filename, 'r'), delimiter=',')
    trans = [map(int, row[1:]) for row in reader]
    return trans


def find_frequent_one(data_set, support):
    """
    Find frequent one itemsets within data set
    :param data_set:
    :param support: Provided support value
    :return:
    """
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


class HNode:
    """
    Class which represents node in a hash tree.
    """

    def __init__(self):
        self.children = {}
        self.isLeaf = True
        self.bucket = {}


class HTree:
    """
    Wrapper class for HTree instance
    """

    def __init__(self, max_leaf_cnt, max_child_cnt):
        self.root = HNode()
        self.max_leaf_cnt = max_leaf_cnt
        self.max_child_cnt = max_child_cnt
        self.frequent_itemsets = []

    def recur_insert(self, node, itemset, index, cnt):
        # TO-DO
        """
        Recursively adds nodes inside the tree and if required splits leaf node and
        redistributes itemsets among child converting itself into intermediate node.
        :param node:
        :param itemset:
        :param index:
        :return:
        """
        if index == len(itemset):
            # last bucket so just insert
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
                # bucket has reached its maximum capacity and its intermediate node so
                # split and redistribute entries.
                for old_itemset, old_cnt in node.bucket.items():

                    hash_key = self.hash_function(old_itemset[index])
                    if hash_key not in node.children:
                        node.children[hash_key] = HNode()
                    self.recur_insert(node.children[hash_key], old_itemset, index + 1, old_cnt)
                # there is no point in having this node's bucket
                # so just delete it
                del node.bucket
                node.isLeaf = False
        else:
            hash_key = self.hash_function(itemset[index])
            if hash_key not in node.children:
                node.children[hash_key] = HNode()
            self.recur_insert(node.children[hash_key], itemset, index + 1, cnt)

    def insert(self, itemset):
        # as list can't be hashed we need to convert this into tuple
        # which can be easily hashed in leaf node buckets
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
                    # print key, value, support_cnt
            return

        for child in node.children.values():
            self.dfs(child, support_cnt)

    def get_frequent_itemsets(self, support_cnt):
        """
        Returns all frequent itemsets which can be considered for next level
        :param support_cnt: Minimum cnt required for itemset to be considered as frequent
        :return:
        """
        self.frequent_itemsets = []
        self.dfs(self.root, support_cnt)
        return self.frequent_itemsets

    def hash_function(self, val):
        return hash(val) % self.max_child_cnt


def generate_hash_tree(candidate_itemsets, length, max_leaf_cnt=4, max_child_cnt=5):
    """
    This function generates hash tree of itemsets with each node having no more than child_max_length
    childs and each leaf node having no more than max_leaf_length.
    :param candidate_itemsets: Itemsets
    :param length: Length if each itemset
    :param max_leaf_length:
    :param child_max_length:
    :return:
    """
    htree = HTree(max_child_cnt, max_leaf_cnt)
    for itemset in candidate_itemsets:
        # add this itemset to hashtree
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
    """
    Generates frequent itemsets
    :param dataset:
    :param support:
    :return: List of f-itemsets with their respective count in
            form of list of tuples.
    """
    support_cnt = int(support / 100.0 * len(dataset))

    # поиск одноэлементных наборов, поддержка которых превышает порог
    all_frequent_itemsets = find_frequent_one(dataset, support)

    prev_frequent = [x[0] for x in all_frequent_itemsets]
    length = 2

    # поиск наборов-кандидатов
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

        # generate hash tree and find frequent itemsets
        h_tree = generate_hash_tree(new_candidates, length)
        # for each transaction, find all possible subsets of size "length"
        k_subsets = generate_k_subsets(dataset, length)

        # support counting and finding frequent itemsets
        for subset in k_subsets:
            h_tree.add_support(subset)

        # find frequent itemsets
        new_frequent = h_tree.get_frequent_itemsets(support_cnt)
        all_frequent_itemsets.extend(new_frequent)
        prev_frequent = [tup[0] for tup in new_frequent]
        prev_frequent.sort()
        length += 1

    return all_frequent_itemsets


def generate_association_rules(f_itemsets, confidence):
    """
    This method generates association rules with confidence greater than threshold
    confidence. For finding confidence we don't need to traverse dataset again as we
    already have support of frequent itemsets.
    Remember Anti-monotone property ?
    I've done pruning in this step also, which reduced its complexity significantly:
    Say X -> Y is AR which don't have enough confidence then any other rule X' -> Y'
    where (X' subset of X) is not possible as sup(X') >= sup(X).
    :param f_itemsets: Frequent itemset with their support values
    :param confidence:
    :return: Returns association rules with associated confidence
    """

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


def create_tmp_rule_table(result_data):
    result_table_name = "pg_apriori_rules_"
    create_table_query = "CREATE TABLE " + result_table_name + \
                         "(" + \
                         "items_from VARCHAR []," + \
                         "items_to VARCHAR []," + \
                         "confidence double precision" + \
                         ")"

    insert_table_query = "INSERT INTO " + result_table_name + \
                         "(items_from, items_to, confidence)" + \
                         " VALUES (ARRAY%s, ARRAY%s, %1.3f)"

    for rule_from, rule_to, confidence in result_data:
        rule_from_string = list(map(lambda r: str(r), rule_from))
        rule_to_string = list(map(lambda r: str(r), rule_to))
        print(insert_table_query % (rule_from_string, rule_to_string, confidence))
    return result_table_name


def print_rules(rules):
    for item in rules:
        left = ','.join(map(str, item[0]))
        right = ','.join(map(str, item[1]))
        print(' ==> '.join([left, right]))
    print('Total Rules Generated: ', len(rules))


def create_tmp_support_table(result_data):
    result_table_name = "pg_apriori_support_"
    create_table_query = "CREATE TABLE " + result_table_name + \
                         "(" + \
                         "items VARCHAR []," + \
                         "support double precision" + \
                         ")"

    insert_table_query = "INSERT INTO " + result_table_name + \
                         "(items, support)" + \
                         " VALUES (ARRAY%s, %1.3f)"

    for item, support in result_data:
        item_string = list(map(lambda r: str(r), item))
        print(insert_table_query % (item_string, support))
    return result_table_name


if __name__ == '__main__':
    dict = {0: ['LBE', '11204', 'Brooklyn'], 1: ['BLACK', 'Cambria Heights', '11411', 'WBE', 'MBE'],
            2: ['Yorktown Heights', '10598', 'BLACK', 'MBE'], 3: ['11561', 'BLACK', 'MBE', 'Long Beach'],
            4: ['11235', 'Brooklyn', 'ASIAN', 'MBE'], 5: ['New York', '10010', 'WBE', 'ASIAN', 'MBE'],
            6: ['10026', 'New York', 'ASIAN', 'MBE'], 7: ['New York', 'BLACK', '10026', 'MBE'],
            8: ['10034', 'New York', 'MBE', 'HISPANIC'], 9: ['BLACK', '10303', 'Staten Island', 'WBE', 'MBE'],
            10: ['10018', 'New York', 'ASIAN', 'MBE'], 11: ['New York', 'HISPANIC', '10034', 'WBE', 'MBE'],
            12: ['New York', 'WBE', 'ASIAN', 'MBE', '10013'], 13: ['Jamaica', 'BLACK', 'MBE', '11434'],
            14: ['NON-MINORITY', 'WBE', 'New York', '10022'], 15: ['10304', 'BLACK', 'MBE', 'Staten Island'],
            16: ['Bronx', 'BLACK', '10454', 'MBE'], 17: ['New Rochelle', 'NON-MINORITY', 'WBE', '10801'],
            18: ['10301', 'NON-MINORITY', 'WBE', 'Staten Island'], 19: ['10006', 'NON-MINORITY', 'WBE', 'New York'],
            20: ['Brooklyn', 'BLACK', '11239', 'MBE'], 21: ['7035', 'Lincoln Park', 'MBE', 'HISPANIC'],
            22: ['BLACK', 'New York', '10027', 'WBE', 'MBE'], 23: ['10310', 'NON-MINORITY', 'WBE', 'Staten Island'],
            24: ['New York', 'ASIAN', 'MBE', '10013'], 25: ['NON-MINORITY', 'Cliffside Park', 'WBE', '7010'],
            26: ['10456', 'Bronx', 'BLACK', 'WBE', 'MBE'], 27: ['LBE', '10003', 'New York'],
            28: ['10303', 'Staten Island', 'MBE', 'HISPANIC'], 29: ['10001', 'New York', 'ASIAN', 'MBE'],
            30: ['New York', '11435', 'BLACK', 'MBE'], 31: ['Ozone Park', 'WBE', '11417'],
            32: ['Lawrence', '11559', 'NON-MINORITY', 'WBE'], 33: ['LBE', 'Brooklyn', '11230', 'ASIAN', 'MBE'],
            34: ['11563', 'Lynbrook', 'MBE', 'HISPANIC'], 35: ['Newark', 'BLACK', 'MBE', '7104'],
            36: ['11356', 'NON-MINORITY', 'WBE', 'College Point'], 37: ['Berkeley Heights', '7922', 'ASIAN', 'MBE'],
            38: ['LBE', 'New York', 'HISPANIC', '10040', 'WBE', 'MBE'], 39: ['East Elmhurst', '11370', 'ASIAN', 'MBE'],
            40: ['LBE', 'Astoria', '11106'], 41: ['MBE', 'New York', 'HISPANIC', 'WBE', '10001'],
            42: ['LBE', 'Bronx', 'BLACK', '10457', 'MBE'], 43: ['South Ozone Park', '11420', 'BLACK', 'WBE', 'MBE'],
            44: ['10920', 'Congers', 'ASIAN', 'MBE'], 45: ['Bronx', '10456', 'BLACK', 'MBE'],
            46: ['11219', 'Brooklyn', 'ASIAN', 'MBE'], 47: ['11360', 'ASIAN', 'MBE', 'Bayside'],
            48: ['10001', 'NON-MINORITY', 'WBE', 'New York'], 49: ['10462', 'Bronx', 'MBE', 'HISPANIC'],
            50: ['LBE', 'Bronx', 'BLACK', '10470', 'MBE'], 51: ['11803', 'Plainview', 'ASIAN', 'MBE'],
            52: ['Bronx', 'NON-MINORITY', 'WBE', '10461'], 53: ['11726', 'NON-MINORITY', 'WBE', 'Copiague'],
            54: ['NON-MINORITY', 'WBE', '98229', 'Bellingham'], 55: ['Jamaica', '11435', 'BLACK', 'MBE'],
            56: ['11375', 'NON-MINORITY', 'WBE', 'Forest Hills'], 57: ['NON-MINORITY', '10012', 'WBE', 'New York'],
            58: ['LBE', 'Bronx', 'BLACK', '10463', 'WBE', 'MBE'], 59: ['11706', 'Bay Shore', 'BLACK', 'MBE'],
            60: ['NON-MINORITY', 'WBE', 'Montclair', '7042'], 61: ['11101', 'NON-MINORITY', 'Elmhurst', 'WBE'],
            62: ['11801', 'ASIAN', 'MBE', 'Hicksville'], 63: ['Westwood', '7675', 'MBE', 'HISPANIC'],
            64: ['10018', 'New York', 'ASIAN', 'MBE'], 65: ['LBE', 'Miller Place', '11764'],
            66: ['Jericho', 'ASIAN', 'MBE', '11753'], 67: ['10107', 'NON-MINORITY', 'WBE', 'New York'],
            68: ['Jamaica', 'BLACK', '11432', 'MBE'], 69: ['Flushing', 'ASIAN', 'MBE', '11367'],
            70: ['BLACK', '10019', 'New York', 'WBE', 'MBE'], 71: ['Richmond Hill', '11418', 'WBE', 'ASIAN', 'MBE'],
            72: ['Hempstead', '11550', 'BLACK', 'MBE'], 73: ['Bronx', 'HISPANIC', '10465', 'WBE', 'MBE'],
            74: ['Croton-on-Hudson', '10520', 'HISPANIC', 'WBE', 'MBE'], 75: ['New York', '10010', 'MBE', 'HISPANIC'],
            76: ['Bronx', 'NON-MINORITY', 'WBE', '10461'], 77: ['NON-MINORITY', 'Mineola', 'WBE', '11501'],
            78: ['10550', 'Mount Vernon', 'ASIAN', 'MBE'], 79: ['10573', 'Port Chester', 'MBE', 'HISPANIC'],
            80: ['NON-MINORITY', 'WBE', '11566', 'Merrick'], 81: ['Long Island City', 'NON-MINORITY', '11101', 'WBE'],
            82: ['11208', 'Brooklyn', 'MBE', 'HISPANIC'], 83: ['11001', 'Floral Park', 'ASIAN', 'MBE'],
            84: ['NON-MINORITY', 'WBE', '10017', 'New York'], 85: ['NON-MINORITY', 'WBE', '10017', 'New York'],
            86: ['10016', 'New York', 'ASIAN', 'MBE'], 87: ['Bronx', '10469', 'BLACK', 'MBE'],
            88: ['10011', 'New York', 'ASIAN', 'MBE'], 89: ['Bronx', '10466', 'BLACK', 'MBE'],
            90: ['East Meadow', 'ASIAN', 'MBE', '11554'], 91: ['NON-MINORITY', '7712', 'WBE', 'Ocean Township'],
            92: ['Bronx', 'ASIAN', 'MBE', '10454'], 93: ['Flushing', 'NON-MINORITY', 'WBE', '11366'],
            94: ['NON-MINORITY', 'WBE', '7004', 'Fairfield'], 95: ['10016', 'New York', 'WBE', 'ASIAN', 'MBE'],
            96: ['Richmond Hill', 'ASIAN', 'MBE', '11418'], 97: ['Brooklyn', 'BLACK', '11216', 'MBE'],
            98: ['New York', 'HISPANIC', 'WBE', 'MBE', '10013'], 99: ['Bronx', '10454', 'HISPANIC', 'WBE', 'MBE'],
            100: ['New York', 'ASIAN', 'MBE', '10010'], 101: ['NON-MINORITY', 'WBE', '7642', 'Hillsdale'],
            102: ['Brooklyn', 'BLACK', '11238', 'WBE', 'MBE'], 103: ['LBE', 'Bayside', '11361', 'ASIAN', 'MBE'],
            104: ['10456', 'Bronx', 'MBE', 'HISPANIC'], 105: ['10018', 'New York', 'ASIAN', 'MBE'],
            106: ['Elmsford', 'NON-MINORITY', '10523', 'WBE'], 107: ['11220', 'Brooklyn', 'ASIAN', 'MBE'],
            108: ['10514', 'NON-MINORITY', 'WBE', 'Chappaqua'], 109: ['Ozone Park', '11417', 'ASIAN', 'MBE'],
            110: ['11435', 'Jamaica', 'ASIAN', 'MBE'], 111: ['Westbury', '11590', 'MBE', 'HISPANIC'],
            112: ['10174', 'New York', 'ASIAN', 'MBE'], 113: ['10032', 'New York', 'MBE', 'HISPANIC'],
            114: ['WBE', '10175', 'New York'], 115: ['10128', 'New York', 'HISPANIC', 'WBE', 'MBE'],
            116: ['11217', 'NON-MINORITY', 'WBE', 'Brooklyn'], 117: ['Brooklyn', 'BLACK', '11225', 'MBE'],
            118: ['Brooklyn', 'BLACK', '11225', 'MBE'], 119: ['WBE', 'HISPANIC', 'Bloomfield', '7003', 'MBE'],
            120: ['10463', 'Bronx', 'MBE', 'HISPANIC'], 121: ['NON-MINORITY', 'Highland Park', 'WBE', '8904'],
            122: ['Astoria', '11102', 'HISPANIC', 'WBE', 'MBE'], 123: ['Brooklyn', '11221', 'BLACK', 'MBE'],
            124: ['NON-MINORITY', '10590', 'WBE', 'South Salem'], 125: ['Bronx', 'BLACK', 'MBE', '10467'],
            126: ['LBE', '10940', 'HISPANIC', 'Middletown', 'MBE'], 127: ['Brooklyn', '11212', 'BLACK', 'MBE'],
            128: ['7060', 'N Plainfield', 'ASIAN', 'MBE'], 129: ['7069', 'BLACK', 'Watchung', 'MBE'],
            130: ['Flushing', '11355', 'ASIAN', 'MBE'], 131: ['Bronx', '10474', 'NON-MINORITY', 'WBE'],
            132: ['11375', 'Forest Hills', 'HISPANIC', 'WBE', 'MBE'], 133: ['Bronx', 'BLACK', 'MBE', '10467'],
            134: ['NON-MINORITY', 'WBE', '10021', 'New York'], 135: ['11357', 'NON-MINORITY', 'WBE', 'Whitestone'],
            136: ['7060', 'BLACK', 'Plainfield', 'MBE'], 137: ['NON-MINORITY', 'WBE', 'New York', '10013'],
            182: ['WBE', 'Brooklyn', 'BLACK', '11234', 'MBE'], 138: ['LBE', 'Brooklyn', '11214', 'ASIAN', 'MBE'],
            139: ['11206', 'NON-MINORITY', 'WBE', 'Brooklyn'], 140: ['LBE', '11434', 'Jamaica', 'HISPANIC', 'MBE'],
            141: ['LIC', 'NON-MINORITY', '11101', 'WBE'], 142: ['7010', 'HISPANIC', 'Cliffside Park', 'WBE', 'MBE'],
            143: ['White Plains', 'NON-MINORITY', '10604', 'WBE'], 144: ['Flushing', 'ASIAN', 'MBE', '11354'],
            145: ['10016', 'New York', 'ASIAN', 'MBE'], 146: ['10314', 'NON-MINORITY', 'WBE', 'Staten Island'],
            147: ['Lindehurst', '11757', 'ASIAN', 'MBE'], 148: ['Jericho', 'ASIAN', 'MBE', '11753'],
            149: ['NON-MINORITY', 'Amityville', 'WBE', '11701'], 150: ['11378', 'Maspeth', 'MBE', 'HISPANIC'],
            151: ['Brooklyn', 'BLACK', '11234', 'WBE', 'MBE'], 152: ['New York', '10030', 'BLACK', 'MBE'],
            153: ['11746', 'Huntington Station', 'BLACK', 'WBE', 'MBE'],
            154: ['NATIVE AMERICAN', 'Lewiston', '14092', 'WBE', 'MBE'], 155: ['10462', 'Bronx', 'MBE', 'HISPANIC'],
            156: ['Far Rockaway', 'BLACK', 'MBE', '11691'], 157: ['St. Albans', 'BLACK', '11412', 'MBE'],
            158: ['NON-MINORITY', 'WBE', 'Brooklyn', '11230'], 159: ['11021', 'Great Neck', 'MBE', 'HISPANIC'],
            160: ['11507', 'Albertson', 'ASIAN', 'MBE'], 161: ['BLACK', '10118', 'New York', 'WBE', 'MBE'],
            162: ['NON-MINORITY', 'WBE', '10017', 'New York'], 163: ['St. Albans', 'BLACK', '11412', 'MBE'],
            164: ['10302', 'Staten Island', 'WBE', 'ASIAN', 'MBE'], 165: ['NON-MINORITY', 'WBE', 'New York', '10010'],
            166: ['NON-MINORITY', 'WBE', 'Flemington', '8822'], 167: ['Brooklyn', 'BLACK', '11203', 'MBE'],
            168: ['7620', 'Alpine', 'WBE', 'ASIAN', 'MBE'], 169: ['10018', 'New York', 'WBE', 'ASIAN', 'MBE'],
            170: ['Hempstead', '11550', 'MBE', 'HISPANIC'], 171: ['10552', 'Mount Vernon', 'MBE', 'HISPANIC'],
            172: ['Queens Village', '11427', 'BLACK', 'MBE'], 173: ['11206', 'NON-MINORITY', 'WBE', 'Brooklyn'],
            174: ['11373', 'Elmhurst', 'MBE', 'HISPANIC'], 175: ['Brooklyn', 'ASIAN', 'MBE', '11232'],
            176: ['Brooklyn', '11231', 'MBE', 'HISPANIC'], 177: ['10306', 'NON-MINORITY', 'WBE', 'Staten Island'],
            178: ['NON-MINORITY', 'WBE', 'New York', '10013'], 179: ['NON-MINORITY', 'WBE', '10021', 'New York'],
            180: ['NON-MINORITY', 'WBE', 'New York', '10018'], 181: ['Brooklyn', 'BLACK', '11205', 'WBE', 'MBE'],
            183: ['South Plainfield', '7080', 'ASIAN', 'MBE'], 184: ['Huntington', 'NON-MINORITY', '11743', 'WBE'],
            185: ['LBE', '11377', 'Woodside', 'ASIAN', 'MBE'], 186: ['10001', 'New York', 'BLACK', 'MBE'],
            187: ['Brooklyn', '11221', 'BLACK', 'MBE'], 188: ['NON-MINORITY', '10176', 'WBE', 'New York'],
            189: ['7601', 'BLACK', 'Hackensack', 'MBE'], 190: ['10462', 'Bronx', 'ASIAN', 'MBE'],
            191: ['10703', 'Yonkers', 'BLACK', 'MBE'], 192: ['Bronx', 'BLACK', 'MBE', '10467'],
            193: ['11040', 'New Hyde Park', 'MBE', 'HISPANIC'], 194: ['East Elmhurst', '11370', 'ASIAN', 'MBE'],
            195: ['New York', '10030', 'BLACK', 'MBE'], 196: ['Long Island City', '11101', 'MBE', 'HISPANIC'],
            197: ['NON-MINORITY', 'WBE', '10302', 'Staten Island'], 198: ['10001', 'NON-MINORITY', 'WBE', 'New York'],
            199: ['MBE', 'New York', 'HISPANIC', 'WBE', '10001'], 200: ['11233', 'Brooklyn', 'MBE', 'HISPANIC'],
            201: ['10165', 'New York', 'ASIAN', 'MBE'], 202: ['Dayton', '8810', 'ASIAN', 'MBE'],
            203: ['11413', 'Springfield Gardens', 'ASIAN', 'MBE'], 204: ['10018', 'New York', 'MBE', 'HISPANIC'],
            205: ['8844', 'ASIAN', 'MBE', 'Hillsborough'], 206: ['Brooklyn', '11226', 'BLACK', 'MBE'],
            207: ['Edison', '8837', 'HISPANIC', 'WBE', 'MBE'], 208: ['10033', 'New York', 'MBE', 'HISPANIC'],
            209: ['Rosedale', '11422', 'BLACK', 'MBE'], 210: ['LBE', '11204', 'Brooklyn', 'ASIAN', 'MBE'],
            211: ['Long Island City', '11101', 'ASIAN', 'MBE'], 212: ['10029', 'BLACK', 'New York', 'WBE', 'MBE'],
            213: ['BLACK', 'Cherry Hill', '8003', 'WBE', 'MBE'],
            214: ['Long Island City', 'NON-MINORITY', '11101', 'WBE'],
            215: ['LBE', '11435', 'Jamaica'], 216: ['NON-MINORITY', 'WBE', '11201', 'Brooklyn'],
            217: ['Woodbridge', '7095', 'ASIAN', 'MBE'], 218: ['11205', 'NON-MINORITY', 'WBE', 'Brooklyn'],
            219: ['Bronx', 'BLACK', '10466', 'WBE', 'MBE'], 220: ['Long Island City', '11106', 'MBE', 'HISPANIC'],
            221: ['10011', 'New York', 'MBE', 'HISPANIC'], 222: ['LBE', 'Brooklyn', '11236', 'ASIAN', 'MBE'],
            223: ['Jamaica', 'BLACK', '11432', 'MBE'], 224: ['Far Rockaway', 'BLACK', 'MBE', '11691'],
            225: ['New York', 'ASIAN', 'MBE', '10013'], 226: ['10011', 'New York', 'ASIAN', 'MBE'],
            227: ['Jamaica', 'BLACK', 'MBE', '11434'], 228: ['New York', '10010', 'MBE', 'HISPANIC'],
            229: ['Union', 'HISPANIC', '7083', 'WBE', 'MBE'], 230: ['Brooklyn', '11208', 'BLACK', 'MBE'],
            231: ['Bronx', 'BLACK', '10454', 'MBE'], 232: ['Morris Plains', '7950', 'WBE', 'ASIAN', 'MBE'],
            233: ['Jackson Heights', '11372', 'MBE', 'HISPANIC'], 234: ['10011', 'New York', 'WBE', 'ASIAN', 'MBE'],
            235: ['10016', 'New York', 'ASIAN', 'MBE'], 236: ['South Nyack', 'BLACK', '10960', 'MBE'],
            237: ['NON-MINORITY', 'WBE', '7511', 'Totowa'], 238: ['7090', 'NON-MINORITY', 'WBE', 'Westfield'],
            239: ['NON-MINORITY', 'WBE', 'New York', '10018'], 240: ['10003', 'NON-MINORITY', 'WBE', 'New York'],
            241: ['11211', 'Brooklyn', 'ASIAN', 'MBE'], 242: ['10469', 'Bronx', 'MBE', 'HISPANIC'],
            243: ['Bronx', 'BLACK', 'MBE', '10470'], 244: ['11430', 'NON-MINORITY', 'WBE', 'Jamaica'],
            245: ['Summit', 'NON-MINORITY', '7901', 'WBE'], 246: ['Bronx', '10460', 'HISPANIC', 'WBE', 'MBE'],
            247: ['10022', 'New York', 'WBE', 'ASIAN', 'MBE'], 248: ['Malverne', 'BLACK', 'MBE', '11565'],
            249: ['LBE', '10012', 'New York', 'ASIAN', 'MBE'], 250: ['11435', 'BLACK', 'MBE', 'Briarwood'],
            251: ['11509', 'Atlantic Beach', 'MBE', 'HISPANIC'], 252: ['11205', 'Brooklyn', 'ASIAN', 'MBE'],
            253: ['Bronx', 'BLACK', '10454', 'MBE'], 254: ['NON-MINORITY', 'WBE', '10941', 'Middletown'],
            255: ['New York', '10030', 'BLACK', 'MBE'], 256: ['BLACK', '10024', 'New York', 'WBE', 'MBE'],
            257: ['11226', 'Brooklyn', 'ASIAN', 'MBE'], 258: ['LBE', '11746', 'Huntington'],
            259: ['Brooklyn', 'BLACK', 'MBE', '11236'], 260: ['10038', 'New York', 'ASIAN', 'MBE'],
            261: ['NON-MINORITY', '7020', 'WBE', 'Edgewater'], 262: ['NON-MINORITY', 'WBE', 'New York', '10010'],
            263: ['10280', 'New York', 'BLACK', 'MBE'], 264: ['Kew Gardens', '11415', 'BLACK', 'MBE'],
            265: ['Bronx', 'BLACK', 'MBE', '10457'], 266: ['Cranbury', '8512', 'ASIAN', 'MBE'],
            267: ['10011', 'New York', 'MBE', 'HISPANIC'], 268: ['Jamaica', '11435', 'BLACK', 'MBE'],
            269: ['Huntington', 'NON-MINORITY', '11743', 'WBE'], 270: ['Ozone Park', '11417', 'ASIAN', 'MBE'],
            271: ['NON-MINORITY', 'WBE', '10025', 'New York'], 272: ['Brooklyn', 'BLACK', '11233', 'WBE', 'MBE'],
            273: ['11218', 'NON-MINORITY', 'WBE', 'Brooklyn'], 274: ['LBE', 'Brooklyn', 'BLACK', '11210', 'MBE'],
            367: ['MBE', 'Bronx', '10466', 'BLACK'], 275: ['10532', 'NON-MINORITY', 'WBE', 'Harrison'],
            276: ['11746', 'NON-MINORITY', 'Dix Hills', 'WBE'], 277: ['NON-MINORITY', '10012', 'WBE', 'New York'],
            278: ['10019', 'New York', 'MBE', 'HISPANIC'], 279: ['Bohemia', 'ASIAN', 'MBE', '11716'],
            280: ['Brooklyn', 'ASIAN', 'MBE', '11232'], 281: ['Island Park', 'NON-MINORITY', 'WBE', '11558'],
            282: ['BLACK', '11001', 'Bellerose Village', 'WBE', 'MBE'], 283: ['BLACK', '11727', 'Coram', 'WBE', 'MBE'],
            284: ['Tarrytown', 'NON-MINORITY', '10591', 'WBE'], 285: ['Brooklyn', 'BLACK', '11224', 'MBE'],
            286: ['NON-MINORITY', '7650', 'WBE', 'Palisades Park'], 287: ['New York', 'BLACK', '10036', 'MBE'],
            288: ['New York', '10019', 'BLACK', 'MBE'], 289: ['10018', 'New York', 'MBE', 'HISPANIC'],
            290: ['NON-MINORITY', '7601', 'WBE', 'Hackensack'], 291: ['10032', 'NON-MINORITY', 'WBE', 'New York'],
            292: ['11757', 'NON-MINORITY', 'WBE', 'Lindenhurst'], 293: ['10012', 'New York', 'MBE', 'HISPANIC'],
            294: ['Westport', 'NON-MINORITY', 'WBE', '6880'], 295: ['10004', 'NON-MINORITY', 'WBE', 'New York'],
            296: ['Nyack', 'NON-MINORITY', 'WBE', '10960'], 297: ['10007', 'NON-MINORITY', 'WBE', 'New York'],
            298: ['BLACK', '10031', 'New York', 'WBE', 'MBE'], 299: ['7087', 'Union City', 'MBE', 'HISPANIC'],
            300: ['7601', 'Hackensack', 'MBE', 'HISPANIC'], 301: ['Flushing', 'ASIAN', 'MBE', '11354'],
            302: ['N/A', 'Syosset', 'WBE', '11791'], 303: ['WBE', 'New York', '10013'],
            304: ['Bronx', '10453', 'BLACK', 'WBE', 'MBE'], 305: ['10023', 'BLACK', 'New York', 'WBE', 'MBE'],
            306: ['Jamaica', 'BLACK', '11432', 'MBE'], 307: ['NON-MINORITY', 'WBE', 'New York', '10013'],
            308: ['10279', 'New York', 'ASIAN', 'MBE'], 309: ['Long Island City', '11106', 'ASIAN', 'MBE'],
            310: ['New York', '10018', 'WBE', 'MBE'], 311: ['11385', 'Ridgewood', 'MBE', 'HISPANIC'],
            312: ['10001', 'New York', 'ASIAN', 'MBE'], 313: ['10310', 'NON-MINORITY', 'WBE', 'Staten Island'],
            314: ['LBE', 'Brooklyn', '11238', 'ASIAN', 'MBE'], 315: ['NON-MINORITY', '10002', 'WBE', 'New York'],
            316: ['10004', 'NON-MINORITY', 'WBE', 'New York'], 317: ['10001', 'New York', 'BLACK', 'MBE'],
            318: ['Elmont', 'BLACK', 'MBE', '11003'], 319: ['10562', 'Ossining', 'BLACK', 'MBE'],
            320: ['New York', '10019', 'BLACK', 'MBE'], 321: ['11501', 'Mineola', 'ASIAN', 'MBE'],
            322: ['NON-MINORITY', 'WBE', 'Valley Stream', '11581'], 323: ['New York', 'BLACK', 'MBE', '10035'],
            324: ['Fort Salonga', 'NON-MINORITY', 'WBE', '11768'], 325: ['11222', 'NON-MINORITY', 'WBE', 'Brooklyn'],
            326: ['LBE', '11040', 'New Hyde Park', 'ASIAN', 'MBE'], 327: ['7070', 'Rutherford', 'WBE', 'ASIAN', 'MBE'],
            328: ['NON-MINORITY', 'WBE', 'Brooklyn', '11209'], 329: ['11556', 'Uniondale', 'BLACK', 'MBE'],
            330: ['LBE', 'Middle Village', '11379'], 331: ['7310', 'Jersey City', 'WBE', 'ASIAN', 'MBE'],
            332: ['10016', 'NON-MINORITY', 'WBE', 'New York'], 333: ['Rosedale', '11422', 'BLACK', 'MBE'],
            334: ['South Richmond Hill', '11419', 'BLACK', 'MBE'], 335: ['NON-MINORITY', 'WBE', '11224', 'Brooklyn'],
            336: ['Brooklyn', 'BLACK', 'MBE', '11236'], 337: ['St. Albans', 'BLACK', '11412', 'MBE'],
            338: ['New York', '10030', 'BLACK', 'MBE'], 339: ['11206', 'Brooklyn', 'MBE', 'HISPANIC'],
            340: ['7405', 'NON-MINORITY', 'WBE', 'Kinnelon'], 341: ['NON-MINORITY', '7601', 'WBE', 'Hackensack'],
            342: ['11729', 'BLACK', 'WBE', 'MBE', 'Deer Park'], 343: ['Bronx', 'NON-MINORITY', 'WBE', '10454'],
            344: ['NON-MINORITY', 'WBE', '11201', 'Brooklyn'], 345: ['New York', '10036', 'MBE', 'HISPANIC'],
            346: ['Ozone Park', '11417', 'ASIAN', 'MBE'], 347: ['10005', 'NON-MINORITY', 'WBE', 'New York'],
            348: ['BLACK', 'Freehold', '7728', 'WBE', 'MBE'], 349: ['NON-MINORITY', 'WBE', 'New York', '10018'],
            350: ['10001', 'NON-MINORITY', 'WBE', 'New York'], 351: ['Old Bethpage', 'NON-MINORITY', '11804', 'WBE'],
            352: ['10456', 'Bronx', 'BLACK', 'WBE', 'MBE'], 353: ['LBE', '8857', 'Old Bridge'],
            354: ['11361', 'ASIAN', 'MBE', 'Bayside'], 355: ['Flushing', '11355', 'ASIAN', 'MBE'],
            356: ['Manasquan', '8736', 'NON-MINORITY', 'WBE'], 357: ['10001', 'New York', 'MBE', 'HISPANIC'],
            358: ['LBE', 'Brooklyn', 'NON-MINORITY', '11229', 'WBE'], 359: ['10465', 'Bronx', 'MBE', 'HISPANIC'],
            360: ['11206', 'NON-MINORITY', 'WBE', 'Brooklyn'], 361: ['New York', 'BLACK', 'MBE', '10035'],
            362: ['NON-MINORITY', 'WBE', 'New York', '10022'], 363: ['11106', 'Astoria', 'MBE', 'HISPANIC'],
            364: ['10004', 'NON-MINORITY', 'WBE', 'New York'], 365: ['Woodside', 'NON-MINORITY', 'WBE', '11377'],
            366: ['11694', 'Rockaway Park', 'MBE', 'HISPANIC'], 368: ['New York', 'BLACK', 'MBE', '10004'],
            369: ['11561', 'Long Beach', 'WBE', 'ASIAN', 'MBE'], 370: ['10120', 'New York', 'ASIAN', 'MBE'],
            371: ['Jamaica', 'BLACK', 'MBE', '11434'], 372: ['BLACK', '10030', 'New York', 'WBE', 'MBE'],
            373: ['BLACK', 'Tenafly', 'MBE', '7670'], 374: ['Brooklyn', 'BLACK', '11210', 'MBE'],
            375: ['10451', 'Bronx', 'MBE', 'HISPANIC'], 376: ['NON-MINORITY', 'WBE', 'New York', '10009'],
            377: ['10016', 'New York', 'ASIAN', 'MBE'], 378: ['LBE', 'Brooklyn', '11231', 'HISPANIC', 'MBE'],
            379: ['10553', 'BLACK', 'MBE', 'Mt. Vernon'], 380: ['LBE', 'Brooklyn', 'BLACK', '11216', 'MBE'],
            381: ['10701', 'Yonkers', 'BLACK', 'MBE'], 382: ['10025', 'New York', 'MBE', 'HISPANIC'],
            383: ['Bronx', 'BLACK', 'MBE', '10457'], 384: ['10312', 'NON-MINORITY', 'WBE', 'Staten Island'],
            385: ['10314', 'NON-MINORITY', 'WBE', 'Staten Island'], 386: ['LBE', '10302', 'Staten Island'],
            387: ['NON-MINORITY', 'WBE', '10594', 'Thornwood'], 388: ['10168', 'New York', 'MBE', 'HISPANIC'],
            389: ['11419', 'Richmond Hill', 'BLACK', 'MBE'], 390: ['Brooklyn', '11218', 'HISPANIC', 'WBE', 'MBE'],
            391: ['10075', 'NON-MINORITY', 'WBE', 'New York'], 392: ['11411', 'BLACK', 'MBE', 'Cambria Heights'],
            393: ['10005', 'NON-MINORITY', 'WBE', 'New York'], 394: ['11722', 'Central Islip', 'MBE', 'HISPANIC'],
            395: ['BLACK', '10026', 'New York', 'WBE', 'MBE'], 396: ['Brooklyn', 'BLACK', '11205', 'WBE', 'MBE'],
            397: ['Bronx', '10475', 'BLACK', 'MBE'], 398: ['New York', '10018', 'BLACK', 'MBE'],
            399: ['LBE', '10314', 'Staten Island'], 400: ['11205', 'Brooklyn', 'ASIAN', 'MBE'],
            401: ['Bronx', '10454', 'HISPANIC', 'WBE', 'MBE'], 402: ['10550', 'Mt. Vernon', 'MBE', 'HISPANIC'],
            403: ['11433', 'Jamaica', 'MBE', 'HISPANIC'], 404: ['New York', 'BLACK', 'MBE', '10004'],
            405: ['Syosset', '11791', 'MBE', 'HISPANIC'], 406: ['10469', 'Bronx', 'MBE', 'HISPANIC'],
            407: ['Brooklyn', '11226', 'BLACK', 'MBE'], 408: ['10993', 'West Haverstraw', 'MBE', 'HISPANIC'],
            409: ['Bronx', 'BLACK', 'WBE', '10458', 'MBE'], 410: ['Brooklyn', 'BLACK', '11216', 'WBE', 'MBE'],
            411: ['NON-MINORITY', '10012', 'WBE', 'New York'],
            412: ['Long Island City', 'NON-MINORITY', '11101', 'WBE'],
            413: ['11501', 'Mineola', 'BLACK', 'MBE'], 414: ['Oakhurst', 'NON-MINORITY', 'WBE', '7755'],
            415: ['10029', 'New York', 'MBE', 'HISPANIC'], 416: ['10038', 'New York', 'MBE', 'HISPANIC'],
            417: ['8812', 'WBE', 'ASIAN', 'MBE', 'Greenbrook'], 418: ['10553', 'BLACK', 'Mount Vernon', 'MBE'],
            419: ['10456', 'Bronx', 'MBE', 'HISPANIC'], 420: ['LBE', 'Astoria', '11106', 'ASIAN', 'MBE'],
            421: ['LBE', 'MBE', 'Roslyn', 'HISPANIC', '11576'], 422: ['Lyndhurst', '7071', 'BLACK', 'MBE'],
            423: ['BLACK', '10031', 'New York', 'WBE', 'MBE'], 424: ['NON-MINORITY', 'WBE', 'New York', '10013'],
            425: ['Brooklyn', '11226', 'BLACK', 'MBE'], 426: ['Bronx', 'BLACK', 'MBE', '10467'],
            427: ['NON-MINORITY', 'WBE', '7645', 'Montvale'], 428: ['NON-MINORITY', '10023', 'WBE', 'New York'],
            429: ['NON-MINORITY', 'WBE', 'Congers', '10920'], 430: ['Brooklyn', 'ASIAN', 'MBE', '11207'],
            431: ['10034', 'New York', 'MBE', 'HISPANIC'], 432: ['NON-MINORITY', 'WBE', 'New York', '10018'],
            433: ['10703', 'Yonkers', 'BLACK', 'MBE'], 434: ['10708', 'NON-MINORITY', 'WBE', 'Bronxville'],
            435: ['10016', 'NON-MINORITY', 'WBE', 'New York'], 436: ['WBE', 'West Islip', '11795', 'ASIAN', 'MBE'],
            437: ['Far Rockaway', '11691', 'MBE', 'HISPANIC'], 438: ['10001', 'NON-MINORITY', 'WBE', 'New York'],
            439: ['Brooklyn', '11205', 'BLACK', 'MBE'], 440: ['7661', 'ASIAN', 'MBE', 'River Edge'],
            441: ['10037', 'BLACK', 'New York', 'WBE', 'MBE'], 442: ['10001', 'New York', 'ASIAN', 'MBE'],
            443: ['Mountainside', 'ASIAN', 'MBE', '7092'], 444: ['Brooklyn', '11220', 'BLACK', 'MBE'],
            445: ['NON-MINORITY', 'WBE', '11215', 'Brooklyn'], 446: ['10024', 'NON-MINORITY', 'WBE', 'New York'],
            447: ['NON-MINORITY', 'WBE', '10017', 'New York'],
            448: ['11101', 'Long Island City', 'WBE', 'ASIAN', 'MBE'],
            449: ['10007', 'New York', 'ASIAN', 'MBE'], 450: ['11208', 'NON-MINORITY', 'WBE', 'Brooklyn'],
            451: ['11385', 'NON-MINORITY', 'WBE', 'Glendale'], 452: ['11218', 'NON-MINORITY', 'WBE', 'Brooklyn'],
            453: ['NON-MINORITY', 'WBE', '11234', 'Brooklyn'], 454: ['10004', 'New York', 'ASIAN', 'MBE'],
            455: ['Bronx', '10455', 'MBE', 'HISPANIC'], 456: ['NON-MINORITY', 'WBE', 'New York', '10010'],
            457: ['LBE', '10475', 'Bronx'], 458: ['Hazlet', 'NON-MINORITY', 'WBE', '7730'],
            459: ['10530', 'BLACK', 'Hartsdale', 'MBE'], 460: ['10306', 'ASIAN', 'MBE', 'Staten Island'],
            461: ['Brooklyn', '11226', 'BLACK', 'MBE'], 462: ['Brooklyn', '11226', 'BLACK', 'MBE'],
            463: ['Hoboken', '7030', 'ASIAN', 'MBE'], 464: ['Ozone Park', '11417', 'ASIAN', 'MBE'],
            465: ['LBE', '11378', 'Maspeth'], 466: ['Ozone Park', '11417', 'MBE', 'HISPANIC'],
            467: ['10310', 'ASIAN', 'MBE', 'Staten Island'], 468: ['NON-MINORITY', 'WBE', '11361', 'Bayside'],
            469: ['Monsey', '10952', 'MBE', 'HISPANIC'], 470: ['11413', 'Springfield Gardens', 'MBE', 'HISPANIC'],
            471: ['Bronx', 'BLACK', 'MBE', '10467'], 472: ['NON-MINORITY', 'WBE', 'New York', '10013'],
            473: ['11560', 'Lattingtown', 'NON-MINORITY', 'WBE'], 474: ['11377', 'Woodside', 'MBE', 'HISPANIC'],
            475: ['Bronx', '10467', 'HISPANIC', 'WBE', 'MBE'], 476: ['BLACK', 'Brookhaven', '11719', 'WBE', 'MBE'],
            477: ['NON-MINORITY', '2738', 'WBE', 'Marion'], 478: ['11232', 'Brooklyn', 'MBE', 'HISPANIC'],
            479: ['Westbury', '11590', 'MBE', 'HISPANIC'], 480: ['10007', 'New York', 'ASIAN', 'MBE'],
            481: ['10304', 'NON-MINORITY', 'WBE', 'Staten Island'], 482: ['Bronx', 'BLACK', 'MBE', '10470'],
            483: ['Howard Beach', 'HISPANIC', '11414', 'WBE', 'MBE'], 484: ['New York', '10018', 'BLACK', 'MBE'],
            485: ['Englewood', 'BLACK', 'MBE', '7631'], 486: ['Brooklyn', 'BLACK', '11216', 'WBE', 'MBE'],
            487: ['11520', 'Freeport', 'BLACK', 'MBE'], 488: ['11722', 'Central Islip', 'MBE', 'HISPANIC'],
            489: ['10003', 'New York', 'HISPANIC', 'WBE', 'MBE'], 490: ['10460', 'Bronx', 'MBE', 'HISPANIC'],
            491: ['11413', 'BLACK', 'Laurelton', 'MBE'], 492: ['Long Island City', '11101', 'MBE'],
            493: ['Flushing', '11354', 'MBE', 'HISPANIC'], 494: ['10550', 'BLACK', 'Mount Vernon', 'MBE'],
            495: ['Brooklyn', 'BLACK', '11210', 'MBE'], 496: ['11208', 'Brooklyn', 'ASIAN', 'MBE'],
            497: ['BLACK', '10026', 'New York', 'WBE', 'MBE'], 498: ['8558', 'Skillman', 'ASIAN', 'MBE'],
            499: ['10001', 'New York', 'ASIAN', 'MBE'], 500: ['10001', 'New York', 'MBE', 'HISPANIC'],
            501: ['Flushing', '11355', 'NON-MINORITY', 'WBE'], 502: ['11237', 'Brooklyn', 'MBE', 'HISPANIC'],
            503: ['BLACK', '10553', 'WBE', 'MBE', 'Mt. Vernon'], 504: ['10001', 'New York', 'MBE', 'HISPANIC'],
            505: ['10016', 'NON-MINORITY', 'WBE', 'New York'], 506: ['Bergenfield', 'NON-MINORITY', 'WBE', '7621'],
            507: ['Teterboro', '7608', 'HISPANIC', 'WBE', 'MBE'], 508: ['NON-MINORITY', '7005', 'WBE', 'Booton'],
            509: ['LBE', '11378', 'Maspeth'], 510: ['Newark', 'BLACK', '7107', 'MBE'],
            511: ['New York', '10038', 'BLACK', 'MBE'], 512: ['11735', 'Farmingdale', 'WBE', 'ASIAN', 'MBE'],
            513: ['Poughquag', '12570', 'ASIAN', 'MBE'], 514: ['Princeton', 'BLACK', 'MBE', '8540'],
            515: ['New York', 'BLACK', 'MBE', '10010'], 516: ['Kenilworth', '7033', 'ASIAN', 'MBE'],
            517: ['10314', 'Staten Island', 'MBE', 'HISPANIC'], 518: ['32963', 'HISPANIC', 'Vero Beach', 'WBE', 'MBE'],
            519: ['Brooklyn', '11205', 'HISPANIC', 'WBE', 'MBE'], 520: ['NON-MINORITY', '10038', 'WBE', 'New York'],
            521: ['10021', 'New York', 'ASIAN', 'MBE'], 522: ['Brooklyn', 'BLACK', 'MBE', '11236'],
            523: ['11206', 'NON-MINORITY', 'WBE', 'Brooklyn'], 524: ['NON-MINORITY', 'Mineola', 'WBE', '11501'],
            525: ['NON-MINORITY', 'WBE', 'New York', '10010'], 526: ['New York', '10029', 'BLACK', 'MBE'],
            527: ['LBE', '10016', 'New York'], 528: ['LBE', 'Maspeth', 'NON-MINORITY', '11378', 'WBE'],
            529: ['10528', 'NON-MINORITY', 'WBE', 'Harrison'], 530: ['ASIAN', 'Medord', 'WBE', '11763', 'MBE'],
            531: ['10001', 'New York', 'BLACK', 'MBE'], 532: ['NON-MINORITY', 'WBE', 'Melville', '11747'],
            533: ['Bethel', '6801', 'BLACK', 'MBE'], 534: ['Long Island City', '11101', 'MBE', 'HISPANIC'],
            535: ['LBE', '11758', 'Massapequa'], 536: ['Newark', 'BLACK', '7107', 'MBE'],
            537: ['NON-MINORITY', '10011', 'WBE', 'New York'], 538: ['Huntington', '11743', 'BLACK', 'MBE'],
            539: ['Baldwin', 'NON-MINORITY', 'WBE', '11510'], 540: ['11411', 'BLACK', 'MBE', 'Cambria Heights'],
            541: ['11375', 'NON-MINORITY', 'WBE', 'Forest Hills'], 542: ['Rego Park', 'BLACK', 'MBE', '11374'],
            543: ['10552', 'BLACK', 'Mount Vernon', 'MBE'], 544: ['NON-MINORITY', 'WBE', '11530', 'Garden City'],
            545: ['11516', 'NON-MINORITY', 'WBE', 'Cedarhurst'], 546: ['NON-MINORITY', '10119', 'WBE', 'New York'],
            547: ['BLACK', 'Elmont', 'WBE', 'MBE', '11003'], 548: ['NON-MINORITY', 'WBE', 'New York', '10010'],
            549: ['Peekskill', '10566', 'MBE', 'HISPANIC'], 550: ['Peekskill', '10566', 'MBE', 'HISPANIC'],
            551: ['NON-MINORITY', 'WBE', '11201', 'Brooklyn'], 552: ['NON-MINORITY', 'Carle Place', 'WBE', '11514'],
            553: ['7601', 'Hackensack', 'HISPANIC', 'WBE', 'MBE'], 554: ['New York', '10010', 'MBE', 'HISPANIC'],
            555: ['Long Island City', 'NON-MINORITY', 'WBE', '11106'], 556: ['11217', 'Brooklyn', 'MBE', 'HISPANIC'],
            557: ['Huntington', '11743', 'BLACK', 'MBE'], 558: ['Flushing', 'ASIAN', 'MBE', '11354'],
            559: ['10005', 'New York', 'MBE', 'HISPANIC'], 560: ['11566', 'Merrick', 'WBE', 'ASIAN', 'MBE'],
            561: ['11566', 'Merrick', 'WBE', 'ASIAN', 'MBE'], 562: ['Brooklyn', '11226', 'BLACK', 'MBE'],
            563: ['NON-MINORITY', 'WBE', '11235', 'Brooklyn'], 564: ['BLACK', '10005', 'New York', 'WBE', 'MBE'],
            565: ['10018', 'New York', 'ASIAN', 'MBE'], 566: ['10461', 'Bronx', 'MBE', 'HISPANIC'],
            567: ['Millstone Township', 'NON-MINORITY', 'WBE', '8535'],
            568: ['Yorktown Heights', 'NON-MINORITY', '10598', 'WBE'],
            569: ['NON-MINORITY', '10011', 'WBE', 'New York'],
            570: ['LBE', '10465', 'Bronx'], 571: ['MBE', 'New York', 'HISPANIC', 'WBE', '10001'],
            572: ['Tuckahoe', '10707', 'MBE', 'HISPANIC'], 573: ['LBE', '10465', 'Bronx'],
            574: ['NON-MINORITY', 'WBE', 'New York', '10010'], 575: ['Brooklyn', '11211', 'BLACK', 'MBE'],
            576: ['Ossining', '10562', 'HISPANIC', 'WBE', 'MBE'], 577: ['White Plains', 'BLACK', '10601', 'WBE', 'MBE'],
            578: ['New York', '10005', 'BLACK', 'MBE'], 579: ['BLACK', '10532', 'Hawthorne', 'WBE', 'MBE'],
            580: ['NON-MINORITY', 'WBE', 'New York', '10013'], 581: ['South Ozone Park', '11420', 'MBE', 'HISPANIC'],
            582: ['Levittown', 'NON-MINORITY', 'WBE', '11756'], 583: ['Bronx', '10451', 'BLACK', 'MBE'],
            584: ['Tinton Falls', 'ASIAN', 'MBE', '7724'], 585: ['NON-MINORITY', 'WBE', 'New York', '10010'],
            586: ['11563', 'BLACK', 'MBE', 'Lynbrook'], 587: ['11237', 'Brooklyn', 'ASIAN', 'MBE'],
            588: ['10005', 'New York', 'MBE', 'HISPANIC'], 589: ['11436', 'Jamaica', 'BLACK', 'MBE'],
            590: ['New York', 'ASIAN', 'MBE', '10013'], 591: ['Brooklyn', 'BLACK', '11216', 'MBE'],
            592: ['11746', 'Dix Hills', 'WBE', 'ASIAN', 'MBE'], 593: ['St. Albans', 'BLACK', '11412', 'MBE'],
            594: ['Hazlet', 'NON-MINORITY', 'WBE', '7730'], 595: ['White Plains', 'BLACK', 'MBE', '10601'],
            596: ['Brooklyn', 'BLACK', '11225', 'MBE'], 597: ['NON-MINORITY', 'WBE', 'New York', '10035'],
            598: ['Brooklyn', '11217', 'BLACK', 'MBE'], 599: ['11580', 'BLACK', 'MBE', 'Valley Stream'],
            600: ['BLACK', '10701', 'Yonkers', 'WBE', 'MBE'], 601: ['BLACK', 'New Rochelle', 'WBE', 'MBE', '10801'],
            602: ['Martinsville', 'NON-MINORITY', 'WBE', '8836'], 603: ['10001', 'NON-MINORITY', 'WBE', 'New York'],
            604: ['NON-MINORITY', 'WBE', '10025', 'New York'], 605: ['NON-MINORITY', 'WBE', '10017', 'New York'],
            606: ['Edgewater', '7020', 'BLACK', 'MBE'], 607: ['11229', 'Brooklyn', 'ASIAN', 'MBE'],
            608: ['East Orange', 'BLACK', '7018', 'MBE'], 609: ['11205', 'Brooklyn', 'ASIAN', 'MBE'],
            610: ['10510', 'Briarcliff Manor', 'ASIAN', 'MBE'], 611: ['Long Island City', '11101', 'MBE', 'HISPANIC'],
            612: ['10003', 'NON-MINORITY', 'WBE', 'New York'], 613: ['Brooklyn', '11212', 'HISPANIC', 'WBE', 'MBE'],
            614: ['NON-MINORITY', 'WBE', 'New York', '10018'], 615: ['10036', 'New York', 'ASIAN', 'MBE'],
            616: ['Rahway', 'NON-MINORITY', 'WBE', '7065'], 617: ['BLACK', '10007', 'New York', 'WBE', 'MBE'],
            618: ['Jamaica', 'BLACK', 'MBE', '11434'], 619: ['10038', 'New York', 'ASIAN', 'MBE'],
            620: ['BLACK', '10026', 'New York', 'WBE', 'MBE'], 621: ['Long Island City', '11101', 'MBE', 'HISPANIC'],
            622: ['10016', 'NON-MINORITY', 'WBE', 'New York'], 623: ['10001', 'NON-MINORITY', 'WBE', 'New York'],
            624: ['Bronx', '10462', 'BLACK', 'MBE'], 625: ['NON-MINORITY', '10012', 'WBE', 'New York'],
            626: ['New York', '10128', 'MBE', 'HISPANIC'], 627: ['Long Island City', '11101', 'MBE', 'HISPANIC'],
            628: ['Brooklyn', 'BLACK', '11213', 'MBE'], 629: ['NON-MINORITY', 'WBE', '11241', 'Brooklyn'],
            630: ['West Babylon', '11704', 'BLACK', 'MBE'], 631: ['10018', 'New York', 'MBE', 'HISPANIC'],
            632: ['Long Island City', 'NON-MINORITY', '11101', 'WBE'],
            633: ['NON-MINORITY', 'WBE', '10017', 'New York'],
            634: ['7652', 'NON-MINORITY', 'WBE', 'Paramus'], 635: ['WBE', '7677', 'Woodcliff Lake', 'ASIAN', 'MBE'],
            636: ['Brooklyn', 'BLACK', '11225', 'MBE'], 637: ['10007', 'NON-MINORITY', 'WBE', 'New York'],
            638: ['LBE', '11101', 'BLACK', 'Long Island City', 'MBE'],
            639: ['HISPANIC', 'Elmont', 'WBE', 'MBE', '11003'],
            640: ['N. Bergen', '7047', 'ASIAN', 'MBE'], 641: ['10005', 'New York', 'MBE', 'HISPANIC'],
            642: ['11704', 'NON-MINORITY', 'WBE', 'West Babylon'], 643: ['10016', 'NON-MINORITY', 'WBE', 'New York'],
            644: ['NON-MINORITY', 'WBE', 'New York', '10165'], 645: ['New York', 'ASIAN', 'MBE', '10022'],
            646: ['10001', 'NON-MINORITY', 'WBE', 'New York'], 647: ['New York', '10006', 'BLACK', 'MBE'],
            648: ['Florham Park', 'BLACK', '7932', 'WBE', 'MBE'], 649: ['Bronx', '10475', 'BLACK', 'MBE'],
            650: ['10001', 'New York', 'ASIAN', 'MBE'], 651: ['Elmhurst', '11373', 'HISPANIC', 'WBE', 'MBE'],
            652: ['10467', 'Bronx', 'ASIAN', 'MBE'], 653: ['WBE', '11201', 'Brooklyn'],
            654: ['10469', 'Bronx', 'ASIAN', 'MBE'], 655: ['10502', 'Ardsley', 'WBE', 'ASIAN', 'MBE'],
            656: ['BLACK', 'New York', '10032', 'WBE', 'MBE'],
            657: ['HISPANIC', 'North Bellmore', '11710', 'WBE', 'MBE'],
            658: ['LBE', '10466', 'Bronx'], 659: ['Bronx', 'BLACK', 'MBE', '10473'],
            660: ['Brooklyn', 'BLACK', '11213', 'MBE'], 661: ['Brooklyn', 'BLACK', 'MBE', '11236'],
            662: ['New York', 'BLACK', 'MBE', '10037'], 663: ['10007', 'New York', 'MBE', 'HISPANIC'],
            664: ['10467', 'Bronx', 'MBE', 'HISPANIC'], 665: ['BLACK', '11747', 'Melville', 'WBE', 'MBE'],
            666: ['Flushing', 'ASIAN', 'MBE', '11354'], 667: ['11216', 'Brooklyn', 'ASIAN', 'MBE'],
            668: ['10030', 'New York', 'MBE', 'HISPANIC'], 669: ['New York', 'BLACK', '10128', 'MBE'],
            670: ['Brooklyn', 'BLACK', 'WBE', '11222', 'MBE'], 671: ['NON-MINORITY', 'WBE', '11201', 'Brooklyn'],
            672: ['New City', '10956', 'HISPANIC', 'WBE', 'MBE'],
            673: ['NON-MINORITY', 'WBE', '11429', 'Queens Village'],
            674: ['Iselin', '8830', 'ASIAN', 'MBE'], 675: ['Jamaica', '11433', 'BLACK', 'MBE'],
            676: ['Queens Village', '11428', 'ASIAN', 'MBE'], 677: ['10002', 'New York', 'WBE', 'ASIAN', 'MBE'],
            678: ['BLACK', '10031', 'New York', 'WBE', 'MBE'], 679: ['Bronx', '10460', 'BLACK', 'MBE'],
            680: ['Roosevelt', '11575', 'MBE', 'HISPANIC'], 681: ['Port Washington', 'ASIAN', 'MBE', '11050'],
            682: ['Brooklyn', '11207', 'MBE', 'HISPANIC'], 683: ['10032', 'NON-MINORITY', 'WBE', 'New York'],
            684: ['Bronx', '10458', 'MBE', 'HISPANIC'], 685: ['Bronx', '10452', 'BLACK', 'MBE'],
            686: ['New York', 'ASIAN', 'MBE', '10010'], 687: ['10550', 'BLACK', 'Mount Vernon', 'MBE'],
            688: ['10471', 'NON-MINORITY', 'WBE', 'Riverdale'], 689: ['Richmond Hills', 'ASIAN', 'MBE', '11418'],
            690: ['Brooklyn', 'BLACK', '11203', 'MBE'], 691: ['7601', 'BLACK', 'Hackensack', 'MBE'],
            692: ['11361', 'ASIAN', 'MBE', 'Bayside'], 693: ['Fairview', '7022', 'MBE', 'HISPANIC'],
            694: ['Flushing', 'ASIAN', 'MBE', '11354'], 695: ['10001', 'New York', 'MBE', 'HISPANIC'],
            696: ['NON-MINORITY', '11731', 'WBE', 'East Northport'], 697: ['11201', 'Brooklyn', 'ASIAN', 'MBE'],
            698: ['Brooklyn', 'BLACK', 'MBE', '11236'], 699: ['Brooklyn', 'BLACK', '11238', 'MBE'],
            700: ['NON-MINORITY', 'WBE', '10017', 'New York'], 701: ['NON-MINORITY', 'WBE', '11201', 'Brooklyn'],
            702: ['Long Island City', '11101', 'BLACK', 'MBE'], 703: ['NON-MINORITY', 'WBE', '11215', 'Brooklyn'],
            704: ['New York', '10027', 'BLACK', 'MBE'], 705: ['LBE', 'Brooklyn', '11225', 'BLACK', 'WBE', 'MBE'],
            706: ['NON-MINORITY', '11729', 'WBE', 'Deer Park'], 707: ['BLACK', '10026', 'New York', 'WBE', 'MBE'],
            708: ['Bohemia', 'BLACK', 'MBE', '11716'], 709: ['LBE', 'Floral Park', '11004', 'ASIAN', 'MBE'],
            710: ['Brooklyn', 'BLACK', '11210', 'MBE'], 711: ['Bronx', '10454', 'MBE', 'HISPANIC'],
            712: ['10301', 'ASIAN', 'MBE', 'Staten Island'], 713: ['Bronx', '10475', 'BLACK', 'MBE'],
            714: ['Bronx', '10475', 'BLACK', 'MBE'], 715: ['Long Island City', '11101', 'BLACK', 'MBE'],
            716: ['10024', 'NON-MINORITY', 'WBE', 'New York'], 717: ['Brooklyn', '11223', 'BLACK', 'WBE', 'MBE'],
            718: ['Mountainside', 'ASIAN', 'MBE', '7092'], 719: ['Ossining', 'NON-MINORITY', 'WBE', '10562'],
            720: ['Westport', 'NON-MINORITY', 'WBE', '6880'],
            721: ['11101', 'Long Island City', 'HISPANIC', 'WBE', 'MBE'],
            722: ['NON-MINORITY', 'WBE', 'New York', '10010'], 723: ['11411', 'BLACK', 'MBE', 'Cambria Heights'],
            724: ['Somerset', 'BLACK', 'MBE', '8873'], 725: ['Millburn', 'NON-MINORITY', 'WBE', '7041'],
            726: ['Baldwin', '11510', 'BLACK', 'MBE'], 727: ['Bronx', 'BLACK', '10466', 'WBE', 'MBE'],
            728: ['NON-MINORITY', 'WBE', 'New York', '10010'], 729: ['Rosedale', '11422', 'BLACK', 'MBE'],
            730: ['10001', 'New York', 'ASIAN', 'MBE'], 731: ['11378', 'Maspeth', 'MBE', 'HISPANIC'],
            732: ['10016', 'NON-MINORITY', 'WBE', 'New York'], 733: ['10280', 'New York', 'HISPANIC', 'WBE', 'MBE'],
            734: ['11357', 'NON-MINORITY', 'WBE', 'Whitestone'],
            735: ['10598', 'BLACK', 'Yorktown Heights', 'WBE', 'MBE'],
            736: ['New York', 'BLACK', 'MBE', '10022'], 737: ['11368', 'Corona', 'ASIAN', 'MBE'],
            966: ['LBE', 'Brooklyn', '11203'], 738: ['NON-MINORITY', 'Holbrook', 'WBE', '11741'],
            739: ['10003', 'NON-MINORITY', 'WBE', 'New York'], 740: ['Brooklyn', 'BLACK', '11205', 'WBE', 'MBE'],
            741: ['10005', 'NON-MINORITY', 'WBE', 'New York'], 742: ['NON-MINORITY', 'WBE', '11201', 'Brooklyn'],
            743: ['Brooklyn', 'BLACK', '11233', 'WBE', 'MBE'], 744: ['Old Bethpage', 'NON-MINORITY', '11804', 'WBE'],
            745: ['10001', 'N/A', 'WBE', 'New York'], 746: ['10007', 'New York', 'ASIAN', 'MBE'],
            747: ['11105', 'Astoria', 'HISPANIC', 'WBE', 'MBE'], 748: ['10037', 'New York', 'MBE', 'HISPANIC'],
            749: ['NON-MINORITY', '11731', 'WBE', 'East Northport'], 750: ['LBE', '10025', 'New York', 'ASIAN', 'MBE'],
            751: ['New York', '10018', 'BLACK', 'MBE'], 752: ['11435', 'NON-MINORITY', 'WBE', 'Briarwood'],
            753: ['10023', 'New York', 'ASIAN', 'MBE'], 754: ['10121', 'New York', 'ASIAN', 'MBE'],
            755: ['BLACK', 'MBE', 'Freeport', 'WBE', '11520'], 756: ['Bronx', '10465', 'NON-MINORITY', 'WBE'],
            757: ['Rosedale', '11422', 'BLACK', 'MBE'], 758: ['7624', 'NON-MINORITY', 'WBE', 'Closter'],
            759: ['10004', 'NON-MINORITY', 'WBE', 'New York'], 760: ['Bronx', '10466', 'BLACK', 'MBE'],
            761: ['Brooklyn', '11207', 'MBE', 'HISPANIC'], 762: ['LBE', '10308', 'Staten Island'],
            763: ['7657', 'Ridgefield', 'ASIAN', 'MBE'], 764: ['Peekskill', '10566', 'ASIAN', 'MBE'],
            765: ['MBE', 'Morristown', 'HISPANIC', 'WBE', '7960'],
            766: ['11042', 'NON-MINORITY', 'WBE', 'Lake Success'],
            767: ['10461', 'Bronx', 'ASIAN', 'MBE'], 768: ['Bronx', 'BLACK', 'MBE', '10461'],
            769: ['10001', 'NON-MINORITY', 'WBE', 'New York'], 770: ['NON-MINORITY', '11211', 'WBE', 'Brooklyn'],
            771: ['West Hempstead', '11552', 'HISPANIC', 'WBE', 'MBE'],
            772: ['10001', 'NON-MINORITY', 'WBE', 'New York'],
            773: ['Brooklyn', 'BLACK', '11203', 'MBE'], 774: ['10012', 'New York', 'WBE', 'ASIAN', 'MBE'],
            775: ['10018', 'New York', 'WBE', 'ASIAN', 'MBE'], 776: ['NON-MINORITY', 'WBE', '11201', 'Brooklyn'],
            777: ['11217', 'NON-MINORITY', 'WBE', 'Brooklyn'], 778: ['7086', 'Weehawken', 'MBE', 'HISPANIC'],
            779: ['6825', 'NON-MINORITY', 'WBE', 'Fairfield'],
            780: ['10583', 'NON-MINORITY', 'Scarsdale', 'WBE', 'MBE'],
            781: ['BLACK', '10701', 'Yonkers', 'WBE', 'MBE'], 782: ['HISPANIC', 'Rego Park', '11374', 'WBE', 'MBE'],
            1102: ['WBE', 'New York', '10025', 'ASIAN', 'MBE'], 783: ['Brooklyn', 'BLACK', '11206', 'WBE', 'MBE'],
            784: ['Long Island City', 'NON-MINORITY', '11101', 'WBE'],
            785: ['7606', 'NON-MINORITY', 'WBE', 'South Hackensack'],
            786: ['Long Island City', '11102', 'ASIAN', 'MBE'],
            787: ['Jackson Heights', '11372', 'ASIAN', 'MBE'], 788: ['10001', 'NON-MINORITY', 'WBE', 'New York'],
            789: ['BLACK', 'Arverne', '11692', 'WBE', 'MBE'], 790: ['NON-MINORITY', 'WBE', '10021', 'New York'],
            791: ['11385', 'Ridgewood', 'MBE', 'HISPANIC'], 792: ['Ozone Park', '11417', 'ASIAN', 'MBE'],
            793: ['NON-MINORITY', '11767', 'WBE', 'Nesconset'], 794: ['NON-MINORITY', 'WBE', 'Roslyn', '11576'],
            795: ['Long Island City', '11101', 'BLACK', 'MBE'], 796: ['10001', 'New York', 'MBE', 'HISPANIC'],
            797: ['10018', 'New York', 'MBE', 'HISPANIC'], 798: ['Westbury', '11590', 'BLACK', 'MBE'],
            799: ['11413', 'BLACK', 'Springfield Gardens', 'MBE'], 800: ['Far Rockaway', 'BLACK', 'MBE', '11691'],
            801: ['10023', 'BLACK', 'New York', 'WBE', 'MBE'], 802: ['10024', 'NON-MINORITY', 'WBE', 'New York'],
            803: ['New York', '10036', 'MBE', 'HISPANIC'], 804: ['Brooklyn', '11216', 'MBE', 'HISPANIC'],
            805: ['Albany', '12205', 'WBE', 'ASIAN', 'MBE'], 806: ['10019', 'New York', 'MBE', 'HISPANIC'],
            807: ['10314', 'NON-MINORITY', 'WBE', 'Staten Island'], 808: ['Brooklyn', 'BLACK', '11234', 'WBE', 'MBE'],
            809: ['11429', 'BLACK', 'MBE', 'WBE', 'Queens Village'], 810: ['Brooklyn', '11221', 'BLACK', 'MBE'],
            811: ['10460', 'Bronx', 'MBE', 'HISPANIC'], 812: ['Brooklyn', 'BLACK', '11205', 'WBE', 'MBE'],
            813: ['Rego Park', 'ASIAN', 'MBE', '11374'], 814: ['Bronx', '10469', 'BLACK', 'MBE'],
            815: ['Hempstead', '11550', 'MBE', 'HISPANIC'], 816: ['11368', 'Corona', 'BLACK', 'MBE'],
            817: ['10467', 'Bronx', 'MBE', 'HISPANIC'], 818: ['LBE', '11206', 'Brooklyn'],
            819: ['LBE', 'Brooklyn', '11210', 'MBE'], 820: ['NON-MINORITY', 'WBE', '11234', 'Brooklyn'],
            821: ['Jackson Heights', '11372', 'MBE', 'HISPANIC'], 822: ['LBE', '10314', 'Staten Island'],
            823: ['11222', 'Brooklyn', 'ASIAN', 'MBE'], 824: ['10032', 'New York', 'MBE', 'HISPANIC'],
            825: ['Brooklyn', 'BLACK', '11234', 'WBE', 'MBE'], 826: ['New York', '10027', 'BLACK', 'MBE'],
            827: ['11563', 'ASIAN', 'MBE', 'Lynbrook'], 828: ['LBE', 'Bronx', 'BLACK', '10469', 'MBE'],
            829: ['Bronx', '10454', 'MBE', 'HISPANIC'], 830: ['11356', 'NON-MINORITY', 'WBE', 'College Point'],
            831: ['8876', 'Branchberg', 'MBE', 'HISPANIC'], 832: ['Brooklyn', '11205', 'BLACK', 'MBE'],
            833: ['Roosevelt', '11575', 'BLACK', 'MBE'], 834: ['7470', 'HISPANIC', 'Wayne', 'WBE', 'MBE'],
            835: ['10710', 'Yonkers', 'BLACK', 'MBE'], 836: ['Mountainside', 'ASIAN', 'MBE', '7092'],
            837: ['Hempstead', '11550', 'ASIAN', 'MBE'], 838: ['NON-MINORITY', 'WBE', '11210', 'Brooklyn'],
            839: ['Brooklyn', 'BLACK', '11201', 'MBE'], 840: ['Jamaica', 'BLACK', 'MBE', '11434'],
            841: ['Richmond Hill', '11418', 'MBE', 'HISPANIC'], 842: ['10031', 'New York', 'MBE', 'HISPANIC'],
            843: ['10028', 'NON-MINORITY', 'WBE', 'New York'], 844: ['11218', 'NON-MINORITY', 'WBE', 'Brooklyn'],
            845: ['NON-MINORITY', '11730', 'WBE', 'East Islip'], 846: ['11751', 'BLACK', 'MBE', 'Islip'],
            847: ['NON-MINORITY', '10038', 'WBE', 'New York'], 848: ['NON-MINORITY', '11767', 'WBE', 'Nesconset'],
            849: ['Flushing', 'ASIAN', 'MBE', '11354'], 850: ['10001', 'NON-MINORITY', 'WBE', 'New York'],
            851: ['11102', 'NON-MINORITY', 'Astoria', 'WBE'], 852: ['New York', '10010', 'MBE', 'HISPANIC'],
            853: ['BLACK', '10031', 'New York', 'WBE', 'MBE'], 854: ['NON-MINORITY', 'WBE', 'New York', '10010'],
            855: ['NON-MINORITY', 'WBE', '10021', 'New York'], 856: ['11791', 'Syosset', 'ASIAN', 'MBE'],
            857: ['10040', 'New York', 'MBE', 'HISPANIC'], 858: ['NON-MINORITY', 'WBE', 'New York', '10018'],
            859: ['LBE', 'Brooklyn', 'BLACK', '11205', 'MBE'], 860: ['New York', 'BLACK', 'MBE', '10031'],
            861: ['East Stroudsburg', '18301', 'BLACK', 'MBE'], 862: ['7728', 'NON-MINORITY', 'WBE', 'Freehold'],
            863: ['11520', 'Freeport', 'BLACK', 'MBE'], 864: ['Flushing', 'NON-MINORITY', 'WBE', '11354'],
            865: ['NON-MINORITY', '11731', 'WBE', 'East Northport'], 866: ['New York', 'WBE', 'ASIAN', 'MBE', '10013'],
            867: ['10303', 'BLACK', 'MBE', 'Staten Island'], 868: ['10022', 'New York', 'MBE', 'HISPANIC'],
            869: ['Brooklyn', '11201', 'WBE', 'ASIAN', 'MBE'], 870: ['NON-MINORITY', 'WBE', 'New York', '10022'],
            871: ['Levittown', '11756', 'MBE', 'HISPANIC'], 872: ['Bronx', 'NON-MINORITY', 'WBE', '10461'],
            873: ['10566', 'BLACK', 'Peeksville', 'WBE', 'MBE'], 874: ['Brooklyn', 'HISPANIC', '11211', 'WBE', 'MBE'],
            1238: ['MBE', '11373', 'Elmhurst', 'ASIAN'], 875: ['HISPANIC', 'Staten Island', '10301', 'WBE', 'MBE'],
            876: ['10701', 'Yonkers', 'BLACK', 'MBE'], 877: ['Richmond Hill', '11418', 'MBE', 'HISPANIC'],
            878: ['7727', 'Farmingdale', 'MBE', 'HISPANIC'], 879: ['Brooklyn', '11208', 'BLACK', 'MBE'],
            880: ['New York', 'ASIAN', 'MBE', '10009'], 881: ['11226', 'Brooklyn', 'ASIAN', 'MBE'],
            882: ['11356', 'College Point', 'MBE', 'HISPANIC'], 883: ['7307', 'ASIAN', 'MBE', 'Jersey City'],
            884: ['10001', 'NON-MINORITY', 'WBE', 'New York'], 885: ['New York', 'HISPANIC', 'WBE', 'MBE', '10033'],
            886: ['11372', 'HISPANIC', 'Jackson Heights', 'WBE', 'MBE'],
            887: ['7002', 'NON-MINORITY', 'WBE', 'Bayonne'],
            888: ['7052', 'West Orange', 'ASIAN', 'MBE'], 889: ['10038', 'New York', 'ASIAN', 'MBE'],
            890: ['11105', 'Astoria', 'MBE', 'HISPANIC'], 891: ['Uniondale', 'NON-MINORITY', '11553', 'WBE'],
            892: ['New York', 'ASIAN', 'MBE', '10122'], 893: ['New York', '10165', 'BLACK', 'MBE'],
            894: ['11746', 'NON-MINORITY', 'Dix Hills', 'WBE'], 895: ['10001', 'New York', 'ASIAN', 'MBE'],
            896: ['10001', 'New York', 'ASIAN', 'MBE'], 897: ['10028', 'NON-MINORITY', 'WBE', 'New York'],
            898: ['11501', 'Mineola', 'BLACK', 'MBE'], 899: ['11220', 'Brooklyn', 'ASIAN', 'MBE'],
            900: ['11422', 'BLACK', 'Rosedale', 'WBE', 'MBE'], 901: ['10006', 'New York', 'ASIAN', 'MBE'],
            902: ['New York', '10003', 'BLACK', 'MBE'], 903: ['North Babylon', 'NON-MINORITY', '11703', 'WBE'],
            904: ['LBE', 'Bronx', 'BLACK', '10466', 'MBE'], 905: ['Scotch Plains', '7076', 'BLACK', 'WBE', 'MBE'],
            906: ['BLACK', 'Maspeth', 'MBE', '11378'], 907: ['NON-MINORITY', 'Albany', 'WBE', '12205'],
            908: ['Jamaica', '11433', 'BLACK', 'MBE'], 909: ['7735', 'ASIAN', 'MBE', 'Cliffwood Beach'],
            910: ['10016', 'NON-MINORITY', 'WBE', 'New York'], 911: ['11501', 'Mineola', 'ASIAN', 'MBE'],
            912: ['Brooklyn', '11211', 'WBE', 'ASIAN', 'MBE'], 913: ['NON-MINORITY', 'WBE', 'New York', '10018'],
            914: ['NON-MINORITY', 'WBE', '11234', 'Brooklyn'], 915: ['10016', 'New York', 'MBE', 'HISPANIC'],
            916: ['Brooklyn', 'BLACK', '11216', 'WBE', 'MBE'], 917: ['10451', 'Bronx', 'MBE', 'HISPANIC'],
            918: ['BLACK', 'Laurelton', '11413', 'WBE', 'MBE'], 919: ['10001', 'NON-MINORITY', 'WBE', 'New York'],
            920: ['Jackson Heights', '11372', 'MBE', 'HISPANIC'], 921: ['7054', 'NON-MINORITY', 'WBE', 'Parsippany'],
            922: ['NON-MINORITY', '11211', 'WBE', 'Brooklyn'], 923: ['NON-MINORITY', 'WBE', '11210', 'Brooklyn'],
            924: ['11357', 'Whitestone', 'WBE', 'ASIAN', 'MBE'], 925: ['11211', 'Brooklyn', 'ASIAN', 'MBE'],
            926: ['MBE', 'Roslyn', 'HISPANIC', 'WBE', '11576'], 927: ['11435', 'NON-MINORITY', 'WBE', 'Jamaica'],
            928: ['Brooklyn', '11212', 'BLACK', 'MBE'], 929: ['Westbury', '11590', 'BLACK', 'MBE'],
            930: ['Brooklyn', 'BLACK', '11212', 'WBE', 'MBE'], 931: ['LBE', '10451', 'Bronx', 'BLACK', 'MBE'],
            932: ['BLACK', '10018', 'New York', 'WBE', 'MBE'], 933: ['10314', 'NON-MINORITY', 'WBE', 'Staten Island'],
            934: ['Brooklyn', '11217', 'BLACK', 'MBE'], 935: ['Hollis', '11423', 'ASIAN', 'MBE'],
            936: ['Brooklyn', 'BLACK', '11210', 'MBE'], 937: ['Bronx', 'BLACK', 'MBE', '10457'],
            938: ['Bronx', '10467', 'HISPANIC', 'WBE', 'MBE'], 939: ['New York', '10118', 'BLACK', 'MBE'],
            940: ['BLACK', 'New York', '10036', 'WBE', 'MBE'], 941: ['Queens', '11411', 'BLACK', 'MBE'],
            942: ['NON-MINORITY', 'WBE', 'Orangeburg', '10962'], 943: ['NON-MINORITY', 'WBE', '11231', 'Brooklyn'],
            944: ['10703', 'Yonkers', 'BLACK', 'MBE'], 945: ['10016', 'New York', 'MBE', 'HISPANIC'],
            946: ['10002', 'New York', 'ASIAN', 'MBE'], 947: ['10701', 'Yonkers', 'BLACK', 'MBE'],
            948: ['Long Island City', '11101', 'ASIAN', 'MBE'], 949: ['11426', 'Bellerose', 'BLACK', 'MBE'],
            950: ['10550', 'BLACK', 'Mount Vernon', 'MBE'], 951: ['10128', 'New York', 'HISPANIC', 'WBE', 'MBE'],
            952: ['Brooklyn', '11205', 'BLACK', 'MBE'], 953: ['LBE', 'Brooklyn', '11218'],
            954: ['11206', 'Brooklyn', 'ASIAN', 'MBE'], 955: ['NON-MINORITY', 'WBE', 'New York', '10013'],
            956: ['NON-MINORITY', 'WBE', '10014', 'New York'], 957: ['NON-MINORITY', 'WBE', '10128', 'New York'],
            958: ['Brooklyn', 'BLACK', '11216', 'WBE', 'MBE'], 959: ['New York', '10018', 'BLACK', 'MBE'],
            960: ['New York', '10030', 'BLACK', 'MBE'], 961: ['10007', 'NON-MINORITY', 'WBE', 'New York'],
            962: ['10001', 'NON-MINORITY', 'WBE', 'New York'], 963: ['10001', 'NON-MINORITY', 'WBE', 'New York'],
            964: ['11725', 'NON-MINORITY', 'Commack', 'WBE'], 965: ['NON-MINORITY', 'WBE', '11238', 'Brooklyn'],
            967: ['White Plains', '10604', 'MBE', 'HISPANIC'], 968: ['Brooklyn', 'ASIAN', 'MBE', '11210'],
            969: ['8837', 'Edison', 'ASIAN', 'MBE'], 970: ['EBE', 'N/A', 'Staten Island', '10301', 'WBE', 'MBE'],
            971: ['Brooklyn', 'BLACK', '11207', 'WBE', 'MBE'], 972: ['Queens Village', 'BLACK', '11429', 'MBE'],
            973: ['10472', 'Bronx', 'ASIAN', 'MBE'], 974: ['Hollis', '11423', 'BLACK', 'MBE'],
            975: ['New York', '10003', 'BLACK', 'MBE'], 976: ['11357', 'BLACK', 'Whitestone', 'MBE'],
            977: ['Brooklyn', '11205', 'BLACK', 'MBE'], 978: ['Brooklyn', '11225', 'BLACK', 'WBE', 'MBE'],
            979: ['10035', 'New York', 'WBE', 'ASIAN', 'MBE'], 980: ['Queens Village', '11427', 'ASIAN', 'MBE'],
            981: ['11782', 'NON-MINORITY', 'WBE', 'Sayville'], 982: ['Great Neck', '11021', 'WBE', 'ASIAN', 'MBE'],
            983: ['11733', 'NON-MINORITY', 'WBE', 'Setauket'], 984: ['10550', 'BLACK', 'Mount Vernon', 'WBE', 'MBE'],
            985: ['10005', 'New York', 'ASIAN', 'MBE'], 986: ['11369', 'E. Elmhurst', 'BLACK', 'MBE'],
            987: ['11040', 'NON-MINORITY', 'WBE', 'New Hyde Park'], 988: ['8861', 'Perth Amboy', 'MBE', 'HISPANIC'],
            989: ['Brooklyn', 'BLACK', '11201', 'MBE'], 990: ['Hollis', '11423', 'BLACK', 'MBE'],
            991: ['10001', 'New York', 'BLACK', 'MBE'], 992: ['BLACK', 'New York', '10025', 'WBE', 'MBE'],
            993: ['BLACK', '10030', 'New York', 'WBE', 'MBE'], 994: ['10003', 'NON-MINORITY', 'WBE', 'New York'],
            995: ['Brooklyn', 'BLACK', '11225', 'MBE'], 996: ['Brooklyn', 'BLACK', '11225', 'MBE'],
            997: ['Wood Ridge', '7075', 'ASIAN', 'MBE'], 998: ['11238', 'Brooklyn', 'ASIAN', 'MBE'],
            999: ['Bronx', 'BLACK', '10454', 'MBE'], 1000: ['NON-MINORITY', 'WBE', '11735', 'Farmingdale'],
            1001: ['LBE', '11356', 'College Point', 'ASIAN', 'MBE'], 1002: ['10003', 'NON-MINORITY', 'WBE', 'New York'],
            1003: ['7446', 'NON-MINORITY', 'WBE', 'Ramsey'], 1004: ['Flushing', 'NON-MINORITY', 'WBE', '11358'],
            1005: ['10475', 'Bronx', 'MBE', 'HISPANIC'], 1006: ['BLACK', 'MBE', 'Freeport', 'WBE', '11520'],
            1007: ['10016', 'NON-MINORITY', 'WBE', 'New York'], 1008: ['Hempstead', '11550', 'BLACK', 'MBE'],
            1009: ['10001', 'NON-MINORITY', 'WBE', 'New York'], 1010: ['10032', 'New York', 'MBE', 'HISPANIC'],
            1011: ['7652', 'NON-MINORITY', 'WBE', 'Paramus'],
            1012: ['Long Island City', 'NON-MINORITY', '11101', 'WBE'],
            1013: ['Brooklyn', 'BLACK', '11210', 'MBE'], 1014: ['11520', 'Freeport', 'BLACK', 'MBE'],
            1015: ['11596', 'Williston Park', 'ASIAN', 'MBE'], 1016: ['11435', 'Jamaica', 'ASIAN', 'MBE'],
            1017: ['LBE', 'Somerville', '8876'], 1018: ['11798', 'BLACK', 'Wheatley Heights', 'MBE'],
            1019: ['11105', 'NON-MINORITY', 'Astoria', 'WBE'], 1020: ['Brooklyn', 'BLACK', '11216', 'MBE'],
            1021: ['NON-MINORITY', '11797', 'WBE', 'Woodbury'], 1022: ['11222', 'Brooklyn', 'ASIAN', 'MBE'],
            1023: ['Jericho', 'ASIAN', 'MBE', '11753'], 1024: ['Westport', 'NON-MINORITY', 'WBE', '6880'],
            1025: ['Baldwin', '11510', 'MBE', 'HISPANIC'], 1026: ['11204', 'Brooklyn', 'ASIAN', 'MBE'],
            1027: ['Brooklyn', '11214', 'WBE', 'ASIAN', 'MBE'], 1028: ['New York', '10019', 'WBE', 'MBE'],
            1029: ['NON-MINORITY', 'WBE', '10021', 'New York'],
            1030: ['LBE', 'BLACK', 'Springfield Gardens', '11413', 'WBE', 'MBE'],
            1031: ['10003', 'NON-MINORITY', 'WBE', 'New York'], 1032: ['BLACK', 'New York', '10036', 'WBE', 'MBE'],
            1033: ['Brooklyn', '11229', 'BLACK', 'MBE'], 1034: ['10005', 'New York', 'MBE', 'HISPANIC'],
            1035: ['10016', 'New York', 'WBE', 'ASIAN', 'MBE'], 1036: ['NON-MINORITY', 'WBE', 'New York', '10010'],
            1037: ['Brooklyn', 'BLACK', '11201', 'MBE'], 1038: ['NON-MINORITY', 'WBE', '10036', 'New York'],
            1039: ['10003', 'NON-MINORITY', 'WBE', 'New York'], 1040: ['10040', 'New York', 'MBE', 'HISPANIC'],
            1041: ['Brooklyn', '11225', 'WBE', 'ASIAN', 'MBE'], 1042: ['NON-MINORITY', '10038', 'WBE', 'New York'],
            1043: ['Summit', 'NON-MINORITY', '7901', 'WBE'], 1044: ['10003', 'New York', 'MBE', 'HISPANIC'],
            1045: ['11501', 'Mineola', 'ASIAN', 'MBE'], 1046: ['10941', 'BLACK', 'MBE', 'Middletown'],
            1047: ['11373', 'Elmhurst', 'MBE', 'HISPANIC'], 1048: ['NON-MINORITY', '10027', 'WBE', 'New York'],
            1049: ['10307', 'ASIAN', 'MBE', 'Staten Island'], 1050: ['NON-MINORITY', 'WBE', 'New York', '10010'],
            1051: ['10977', 'BLACK', 'Chestnut Ridge', 'WBE', 'MBE'],
            1052: ['BLACK', '10007', 'New York', 'WBE', 'MBE'],
            1053: ['Rosedale', '11422', 'BLACK', 'MBE'], 1054: ['Garden City Park', '11040', 'WBE', 'ASIAN', 'MBE'],
            1055: ['10550', 'BLACK', 'MBE', 'Mt. Vernon'], 1056: ['11040', 'NON-MINORITY', 'WBE', 'New Hyde Park'],
            1057: ['Long Island City', 'NON-MINORITY', '11101', 'WBE'], 1058: ['New York', 'ASIAN', 'MBE', '10013'],
            1059: ['10018', 'New York', 'WBE', 'ASIAN', 'MBE'], 1060: ['New York', 'BLACK', '10017', 'MBE'],
            1061: ['10018', 'New York', 'MBE', 'HISPANIC'], 1062: ['LBE', 'Brooklyn', 'BLACK', '11234', 'MBE'],
            1063: ['Princeton', 'ASIAN', 'MBE', '8540'], 1064: ['BLACK', 'New York', '10027', 'WBE', 'MBE'],
            1065: ['7003', 'NON-MINORITY', 'WBE', 'Bloomfield'], 1066: ['NON-MINORITY', 'WBE', '11697', 'Breezy Point'],
            1067: ['Yonkers', '10705', 'BLACK', 'MBE'], 1068: ['10462', 'Bronx', 'HISPANIC', 'WBE', 'MBE'],
            1069: ['10977', 'NON-MINORITY', 'WBE', 'New Hempstead'], 1070: ['New York', '10027', 'BLACK', 'MBE'],
            1071: ['New York', '10038', 'WBE', 'ASIAN', 'MBE'], 1072: ['Flushing', 'BLACK', 'MBE', '11354'],
            1073: ['10460', 'Bronx', 'MBE', 'HISPANIC'], 1074: ['10304', 'BLACK', 'MBE', 'Staten Island'],
            1075: ['NON-MINORITY', 'WBE', '8816', 'East Brunswick'], 1076: ['NON-MINORITY', '11211', 'WBE', 'Brooklyn'],
            1077: ['NON-MINORITY', 'WBE', 'New York', '10022'], 1078: ['LBE', 'Bronx', 'BLACK', '10467', 'MBE'],
            1079: ['10038', 'New York', 'ASIAN', 'MBE'], 1080: ['7601', 'Hackensack', 'WBE', 'ASIAN', 'MBE'],
            1081: ['10005', 'New York', 'ASIAN', 'MBE'], 1082: ['10470', 'Bronx', 'ASIAN', 'MBE'],
            1083: ['11385', 'ASIAN', 'MBE', 'Glendale'], 1084: ['11803', 'Plainview', 'ASIAN', 'MBE'],
            1085: ['NON-MINORITY', 'WBE', '10014', 'New York'], 1086: ['NON-MINORITY', 'WBE', 'Allendale', '7401'],
            1087: ['10314', 'ASIAN', 'MBE', 'Staten Island'], 1088: ['Brooklyn', 'BLACK', '11214', 'WBE', 'MBE'],
            1089: ['7090', 'NON-MINORITY', 'WBE', 'Westfield'], 1090: ['Edison', 'WBE', 'ASIAN', '8817'],
            1091: ['NON-MINORITY', 'WBE', '11572', 'Oceanside'], 1092: ['Edison', 'NON-MINORITY', 'WBE', '8837'],
            1093: ['NON-MINORITY', 'WBE', '10025', 'New York'], 1094: ['11357', 'NON-MINORITY', 'WBE', 'Whitestone'],
            1095: ['NON-MINORITY', 'WBE', '11238', 'Brooklyn'], 1096: ['10022', 'New York', 'HISPANIC', 'WBE', 'MBE'],
            1097: ['BLACK', 'Cambria Heights', '11411', 'WBE', 'MBE'],
            1098: ['10029', 'BLACK', 'New York', 'WBE', 'MBE'],
            1099: ['MBE', 'Freeport', 'HISPANIC', 'WBE', '11520'], 1100: ['Bronx', '10474', 'BLACK', 'MBE'],
            1101: ['10001', 'NON-MINORITY', 'WBE', 'New York'], 1103: ['Brooklyn', 'BLACK', 'WBE', '11210', 'MBE'],
            1104: ['10118', 'New York', 'ASIAN', 'MBE'], 1105: ['10473', 'Bronx', 'MBE', 'HISPANIC'],
            1106: ['10451', 'Bronx', 'BLACK', 'WBE', 'MBE'], 1107: ['10470', 'Bronx', 'MBE', 'HISPANIC'],
            1108: ['BLACK', 'Yonkers', '10710', 'WBE', 'MBE'], 1109: ['10301', 'NON-MINORITY', 'WBE', 'Staten Island'],
            1110: ['NON-MINORITY', '10011', 'WBE', 'New York'], 1111: ['Woodside', 'NON-MINORITY', 'WBE', '11377'],
            1112: ['10004', 'New York', 'ASIAN', 'MBE'], 1113: ['NON-MINORITY', 'WBE', '10017', 'New York'],
            1114: ['8837', 'Edison', 'ASIAN', 'MBE'], 1115: ['NON-MINORITY', 'WBE', 'New York', '10022'],
            1116: ['NON-MINORITY', 'WBE', '10017', 'New York'], 1117: ['10022', 'New York', 'HISPANIC', 'WBE', 'MBE'],
            1118: ['LBE', 'Elmhurst', '11373', 'ASIAN', 'MBE'], 1119: ['8817', 'Edsion', 'ASIAN', 'MBE'],
            1120: ['LBE', 'BLACK', '10304', 'Staten Island', 'MBE'],
            1121: ['Queens Village', '11429', 'MBE', 'HISPANIC'],
            1122: ['New York', '10010', 'HISPANIC', 'WBE', 'MBE'],
            1123: ['LBE', '11377', 'Woodside', 'HISPANIC', 'MBE'],
            1124: ['LBE', 'BLACK', 'St. Albans', '11412', 'MBE'], 1125: ['11378', 'Maspeth', 'MBE', 'HISPANIC'],
            1126: ['Maspeth', '11378', 'HISPANIC', 'WBE', 'MBE'], 1127: ['10003', 'NON-MINORITY', 'WBE', 'New York'],
            1128: ['11042', 'Lake Success', 'ASIAN', 'MBE'], 1129: ['10016', 'WBE', 'New York'],
            1130: ['New York', 'WBE', 'ASIAN', 'MBE', '10013'], 1131: ['NON-MINORITY', 'WBE', '10305', 'Staten Island'],
            1132: ['NON-MINORITY', 'Plainview', 'WBE', '11803'], 1133: ['11704', 'NON-MINORITY', 'WBE', 'West Babylon'],
            1134: ['NON-MINORITY', 'WBE', 'New York', '10013'], 1135: ['11430', 'NON-MINORITY', 'WBE', 'Jamaica'],
            1136: ['Bronx', '10462', 'BLACK', 'MBE'], 1137: ['NON-MINORITY', '10023', 'WBE', 'New York'],
            1138: ['Saddle Brook', 'WBE', 'ASIAN', '7663'], 1139: ['NON-MINORITY', 'WBE', '11694', 'Belle Harbor'],
            1140: ['LBE', 'Brooklyn', 'BLACK', '11205', 'MBE'], 1141: ['11208', 'Brooklyn', 'ASIAN', 'MBE'],
            1142: ['Yorktown Heights', 'NON-MINORITY', '10598', 'WBE'], 1143: ['LBE', '10451', 'Bronx', 'BLACK', 'MBE'],
            1144: ['Linwood', 'ASIAN', 'MBE', '8221'], 1145: ['10591', 'Sleepy Hollow', 'WBE', 'ASIAN', 'MBE'],
            1146: ['Centerport', 'HISPANIC', '11721', 'WBE', 'MBE'], 1147: ['NON-MINORITY', 'WBE', 'Melville', '11747'],
            1148: ['10004', 'NON-MINORITY', 'WBE', 'New York'], 1149: ['NON-MINORITY', 'WBE', 'New York', '10018'],
            1150: ['LBE', '10304', 'Staten Island'], 1151: ['New York', '10018', 'BLACK', 'MBE'],
            1152: ['Queens Village', '11428', 'ASIAN', 'MBE'],
            1153: ['11746', 'NON-MINORITY', 'WBE', 'Huntington Station'],
            1154: ['11206', 'Brooklyn', 'ASIAN', 'MBE'], 1155: ['New York', '10018', 'BLACK', 'MBE'],
            1156: ['Bronx', '10466', 'HISPANIC', 'WBE', 'MBE'], 1157: ['NON-MINORITY', 'WBE', '11735', 'Farmingdale'],
            1158: ['Flushing', 'NON-MINORITY', 'WBE', '11358'], 1159: ['NON-MINORITY', 'WBE', '11234', 'Brooklyn'],
            1160: ['BLACK', '10005', 'New York', 'WBE', 'MBE'], 1161: ['10001', 'NON-MINORITY', 'WBE', 'New York'],
            1162: ['11021', 'NON-MINORITY', 'WBE', 'Great Neck'], 1163: ['10001', 'New York', 'BLACK', 'MBE'],
            1164: ['NON-MINORITY', 'WBE', 'New York', '10018'], 1165: ['11422', 'BLACK', 'Rosedale', 'WBE', 'MBE'],
            1166: ['11242', 'NON-MINORITY', 'WBE', 'Brooklyn'], 1167: ['Brooklyn', 'BLACK', '11225', 'MBE'],
            1168: ['Hollis', '11423', 'BLACK', 'MBE'], 1169: ['Brooklyn', 'BLACK', '11216', 'MBE'],
            1170: ['NON-MINORITY', 'WBE', 'Dobbs Ferry', '10522'], 1171: ['NON-MINORITY', 'WBE', 'New York', '10010'],
            1172: ['NON-MINORITY', 'WBE', 'New York', '10018'], 1173: ['Brooklyn', 'BLACK', '11230', 'WBE', 'MBE'],
            1174: ['NON-MINORITY', 'WBE', '10025', 'New York'], 1175: ['Brooklyn', 'BLACK', '11203', 'MBE'],
            1176: ['10514', 'NON-MINORITY', 'WBE', 'Chappaqua'], 1177: ['Stamford', 'NON-MINORITY', 'WBE', '6902'],
            1178: ['11545', 'Glen Head', 'BLACK', 'MBE'], 1179: ['Brooklyn', 'BLACK', '11216', 'WBE', 'MBE'],
            1180: ['10458', 'Bronx', 'ASIAN', 'MBE'], 1181: ['10001', 'NON-MINORITY', 'WBE', 'New York'],
            1182: ['East Elmhurst', '11370', 'ASIAN', 'MBE'], 1183: ['10001', 'New York', 'ASIAN', 'MBE'],
            1184: ['LBE', 'Hempstead', 'BLACK', '11550', 'WBE', 'MBE'], 1185: ['Brooklyn', '11204', 'BLACK', 'MBE'],
            1186: ['LBE', 'Maspeth', '11378', 'HISPANIC', 'WBE', 'MBE'],
            1187: ['BLACK', '20720', 'Bowie', 'WBE', 'MBE'],
            1188: ['11746', 'NON-MINORITY', 'WBE', 'Huntington Station'],
            1189: ['Long Island City', '11101', 'BLACK', 'MBE'], 1190: ['Woodhaven', '11421', 'BLACK', 'MBE'],
            1191: ['New York', 'BLACK', 'MBE', '10033'], 1192: ['NON-MINORITY', 'WBE', 'Valhalla', '10595'],
            1193: ['New York', 'WBE', 'MBE', '10032'], 1194: ['7513', 'Paterson', 'MBE', 'HISPANIC'],
            1195: ['11356', 'College Point', 'MBE', 'HISPANIC'], 1196: ['NON-MINORITY', 'Springfield', 'WBE', '7081'],
            1197: ['NON-MINORITY', 'WBE', '11216', 'Brooklyn'], 1198: ['New York', '10038', 'BLACK', 'MBE'],
            1199: ['Matawan', '7747', 'MBE', 'HISPANIC'], 1200: ['Newark', 'NON-MINORITY', 'WBE', '7114'],
            1201: ['8638', 'Hamilton Township', 'ASIAN', 'MBE'], 1202: ['11021', 'NON-MINORITY', 'WBE', 'Great Neck'],
            1203: ['Long Island City', '11101', 'MBE', 'HISPANIC'], 1204: ['10016', 'NON-MINORITY', 'WBE', 'New York'],
            1205: ['Bronx', 'HISPANIC', '10465', 'WBE', 'MBE'], 1206: ['10025', 'New York', 'MBE', 'HISPANIC'],
            1207: ['NON-MINORITY', 'WBE', '10021', 'New York'], 1208: ['11373', 'Elmhurst', 'ASIAN', 'MBE'],
            1209: ['10016', 'New York', 'WBE', 'ASIAN', 'MBE'], 1210: ['NON-MINORITY', '10956', 'WBE', 'New City'],
            1211: ['10022', 'New York', 'MBE', 'HISPANIC'], 1212: ['NON-MINORITY', 'WBE', 'New York', '10010'],
            1213: ['10543', 'Mamaroneck', 'BLACK', 'MBE'], 1214: ['BLACK', 'Arverne', '11692', 'WBE', 'MBE'],
            1215: ['10004', 'NON-MINORITY', 'WBE', 'New York'], 1216: ['7045', 'Montville', 'MBE', 'HISPANIC'],
            1217: ['10028', 'NON-MINORITY', 'WBE', 'New York'], 1218: ['NON-MINORITY', 'WBE', '11201', 'Brooklyn'],
            1219: ['NON-MINORITY', 'WBE', '10017', 'New York'], 1220: ['NON-MINORITY', '11590', 'WBE', 'Westbury'],
            1221: ['Brooklyn', '11225', 'MBE', 'HISPANIC'], 1222: ['10007', 'New York', 'BLACK', 'MBE'],
            1223: ['Bronx', '10451', 'BLACK', 'MBE'], 1224: ['St. Albans', 'BLACK', '11412', 'MBE'],
            1225: ['Brooklyn', 'BLACK', '11236', 'WBE', 'MBE'], 1226: ['11218', 'Brooklyn', 'ASIAN', 'MBE'],
            1227: ['Brooklyn', 'BLACK', '11234', 'WBE', 'MBE'], 1228: ['11218', 'Brooklyn', 'ASIAN', 'MBE'],
            1229: ['Brooklyn', 'BLACK', '11230', 'WBE', 'MBE'], 1230: ['NON-MINORITY', '10019', 'WBE', 'New York'],
            1231: ['12534', 'NON-MINORITY', 'WBE', 'Hudson'], 1232: ['7876', 'NON-MINORITY', 'WBE', 'Succasunna'],
            1233: ['New York', '10038', 'BLACK', 'MBE'], 1234: ['NON-MINORITY', 'WBE', '11231', 'Brooklyn'],
            1235: ['Scarsdale', 'NON-MINORITY', 'WBE', '10583'], 1236: ['11413', 'BLACK', 'Laurelton', 'MBE'],
            1237: ['11223', 'Brooklyn', 'ASIAN', 'MBE'], 1239: ['Bronx', 'BLACK', '10467', 'WBE', 'MBE'],
            1240: ['10462', 'Bronx', 'BLACK', 'WBE', 'MBE'], 1241: ['10309', 'NON-MINORITY', 'WBE', 'Staten Island'],
            1242: ['New York', 'ASIAN', 'MBE', '10010'], 1243: ['New York', '10014', 'MBE', 'HISPANIC'],
            1244: ['Hollis', '11423', 'N/A', 'MBE'], 1245: ['NON-MINORITY', 'WBE', '10036', 'New York'],
            1246: ['Southold', 'NON-MINORITY', 'WBE', '11971'],
            1247: ['LBE', '11746', 'Huntington Station', 'BLACK', 'MBE'],
            1248: ['LBE', '10039', 'New York', 'ASIAN', 'MBE'], 1249: ['10710', 'Yonkers', 'MBE', 'HISPANIC'],
            1250: ['Hempstead', '11550', 'BLACK', 'MBE'], 1251: ['11419', 'Richmond Hill', 'ASIAN', 'MBE'],
            1252: ['11217', 'NON-MINORITY', 'WBE', 'Brooklyn'], 1253: ['Yonkers', '10710', 'NON-MINORITY', 'WBE'],
            1254: ['Woodside', 'ASIAN', 'MBE', '11377'], 1255: ['Edison', '8820', 'ASIAN', 'MBE'],
            1256: ['10016', 'NON-MINORITY', 'WBE', 'New York'], 1257: ['NON-MINORITY', 'WBE', '10014', 'New York'],
            1258: ['10018', 'New York', 'WBE', 'ASIAN', 'MBE'], 1259: ['NON-MINORITY', 'WBE', '10036', 'New York'],
            1260: ['North Babylon', 'NON-MINORITY', '11704', 'WBE'], 1261: ['10018', 'New York', 'WBE', 'ASIAN', 'MBE'],
            1262: ['New York', '10018', 'BLACK', 'MBE'], 1263: ['6478', 'Oxford', 'ASIAN', 'MBE'],
            1264: ['LBE', 'Brooklyn', '11234'], 1265: ['10001', 'NON-MINORITY', 'WBE', 'New York'],
            1266: ['New York', '10010', 'MBE', 'HISPANIC'], 1267: ['10301', 'ASIAN', 'MBE', 'Staten Island'],
            1268: ['White Plains', '10604', 'WBE'], 1269: ['10003', 'New York', 'ASIAN', 'MBE'],
            1270: ['Bronx', '10453', 'BLACK', 'MBE'], 1271: ['BLACK', 'New York', '10034', 'WBE', 'MBE'],
            1272: ['7430', 'NON-MINORITY', 'WBE', 'Mahwah'], 1273: ['Inwood', '11096', 'HISPANIC', 'WBE', 'MBE'],
            1274: ['10005', 'NON-MINORITY', 'WBE', 'New York'], 1275: ['10001', 'New York', 'ASIAN', 'MBE'],
            1276: ['Brooklyn', 'BLACK', 'MBE', '11233'], 1277: ['Hollis', '11423', 'BLACK', 'MBE'],
            1278: ['Bronx', '10452', 'BLACK', 'MBE'], 1279: ['10038', 'New York', 'ASIAN', 'MBE'],
            1280: ['NON-MINORITY', 'WBE', 'New York', '10022'], 1281: ['10017', 'New York', 'ASIAN', 'MBE'],
            1282: ['Bronx', '10469', 'BLACK', 'MBE'], 1283: ['10550', 'BLACK', 'Mount Vernon', 'MBE'],
            1284: ['NON-MINORITY', '10011', 'WBE', 'New York'], 1285: ['NON-MINORITY', 'WBE', 'New York', '10018'],
            1286: ['NON-MINORITY', 'WBE', 'Brooklyn', '11232'],
            1287: ['BLACK', 'Springfield Gardens', '11413', 'WBE', 'MBE'], 1288: ['Flushing', '11358', 'ASIAN', 'MBE'],
            1289: ['10003', 'NON-MINORITY', 'WBE', 'New York'], 1290: ['Woodside', 'ASIAN', 'MBE', '11377'],
            1291: ['NON-MINORITY', '11729', 'WBE', 'Deer Park'], 1292: ['11206', 'Brooklyn', 'MBE', 'HISPANIC'],
            1293: ['Bronx', 'BLACK', 'MBE', '10457'], 1294: ['LBE', 'Brooklyn', '11235', 'ASIAN', 'MBE'],
            1295: ['NON-MINORITY', 'WBE', '11215', 'Brooklyn'], 1296: ['Bronx', 'BLACK', '10466', 'WBE', 'MBE'],
            1297: ['Hempstead', '11520', 'BLACK', 'MBE'], 1298: ['10006', 'New York', 'ASIAN', 'MBE'],
            1299: ['Rosedale', '11422', 'BLACK', 'MBE'], 1300: ['Flushing', '11358', 'MBE', 'HISPANIC'],
            1301: ['Edison', '8817', 'ASIAN', 'MBE'], 1302: ['Brooklyn', '11205', 'BLACK', 'MBE'],
            1303: ['Rahway', 'BLACK', '7065', 'MBE'], 1304: ['NON-MINORITY', 'WBE', 'New York', '10018'],
            1305: ['11413', 'BLACK', 'Springfield Gardens', 'MBE'], 1306: ['Bronx', '10472', 'BLACK', 'MBE'],
            1307: ['11211', 'Brooklyn', 'MBE', 'HISPANIC'], 1308: ['NON-MINORITY', 'WBE', '10025', 'New York'],
            1309: ['LBE', '11378', 'Maspeth'], 1310: ['10031', 'New York', 'HISPANIC', 'WBE', 'MBE'],
            1311: ['7031', 'NON-MINORITY', 'North Arlington', 'WBE'], 1312: ['Saugerties', '12477', 'WBE'],
            1313: ['Queens Village', '11428', 'BLACK', 'MBE'], 1314: ['10017', 'New York', 'ASIAN', 'MBE'],
            1315: ['11732', 'East Norwich', 'ASIAN', 'MBE'], 1316: ['Brooklyn', 'BLACK', 'MBE', '11236'],
            1317: ['10016', 'NON-MINORITY', 'WBE', 'New York'], 1318: ['7003', 'Bloomfield', 'BLACK', 'MBE'],
            1319: ['NON-MINORITY', '10011', 'WBE', 'New York'],
            1320: ['NON-MINORITY', 'WBE', 'Center Moriches', '11934'],
            1321: ['10016', 'NON-MINORITY', 'WBE', 'New York'], 1322: ['10001', 'New York', 'ASIAN', 'MBE'],
            1323: ['10701', 'Yonkers', 'BLACK', 'MBE'], 1324: ['Brooklyn', 'BLACK', '11203', 'MBE'],
            1325: ['Long Island City', 'NON-MINORITY', '11101', 'WBE'],
            1326: ['Staten Island', '10306', 'WBE', 'ASIAN', 'MBE'], 1327: ['Brooklyn', 'BLACK', 'MBE', '11233'],
            1328: ['Bronx', '10471', 'NON-MINORITY', 'WBE'], 1329: ['10460', 'Bronx', 'MBE', 'HISPANIC'],
            1330: ['10011', 'New York', 'HISPANIC', 'WBE', 'MBE'], 1331: ['New York', '10010', 'MBE', 'HISPANIC'],
            1332: ['New York', 'HISPANIC', '10040', 'WBE', 'MBE'], 1333: ['11104', 'NON-MINORITY', 'WBE', 'Sunnyside'],
            1334: ['BLACK', 'Rego Park', '11374', 'WBE', 'MBE'], 1335: ['10280', 'New York', 'HISPANIC', 'WBE', 'MBE'],
            1336: ['10016', 'NON-MINORITY', 'WBE', 'New York'], 1337: ['BLACK', '11435', 'Jamaica', 'WBE', 'MBE'],
            1338: ['21207', 'BLACK', 'Baltimore', 'MBE'], 1339: ['Brooklyn', 'BLACK', '11201', 'MBE'],
            1340: ['Brooklyn', 'BLACK', 'MBE', '11236'], 1341: ['NON-MINORITY', 'WBE', 'Brooklyn', '11230'],
            1342: ['BLACK', '8844', 'MBE', 'Hillsborough'], 1343: ['10001', 'New York', 'BLACK', 'MBE'],
            1344: ['Floral Park', '11001', 'HISPANIC', 'WBE', 'MBE'],
            1345: ['10001', 'NON-MINORITY', 'WBE', 'New York'],
            1346: ['11413', 'BLACK', 'Laurelton', 'MBE'], 1347: ['NON-MINORITY', 'WBE', '10036', 'New York'],
            1348: ['10308', 'NON-MINORITY', 'WBE', 'Staten Island'],
            1349: ['10011', 'New York', 'HISPANIC', 'WBE', 'MBE'],
            1350: ['7751', 'HISPANIC', 'Morganville', 'WBE', 'MBE'], 1351: ['NON-MINORITY', 'WBE', 'New York', '10018'],
            1352: ['Jamaica', 'BLACK', 'MBE', '11434'], 1353: ['10004', 'New York', 'WBE', 'ASIAN', 'MBE'],
            1354: ['10016', 'NON-MINORITY', 'WBE', 'New York'], 1355: ['10012', 'New York', 'HISPANIC', 'WBE', 'MBE'],
            1356: ['7205', 'Hillside', 'HISPANIC', 'WBE', 'MBE'], 1357: ['10018', 'New York', 'WBE', 'ASIAN', 'MBE'],
            1358: ['NON-MINORITY', '10023', 'WBE', 'New York'], 1359: ['New York', 'BLACK', '10128', 'MBE'],
            1360: ['NON-MINORITY', '10038', 'WBE', 'New York'], 1361: ['11520', 'Freeport', 'BLACK', 'MBE'],
            1362: ['WBE', 'ASIAN', 'New York', '10013'], 1363: ['Bronx', '10453', 'BLACK', 'WBE', 'MBE'],
            1364: ['South Richmond Hill', '11419', 'ASIAN', 'MBE'],
            1365: ['South Richmond Hill', '11419', 'BLACK', 'MBE'],
            1366: ['NON-MINORITY', 'WBE', '11020', 'Great Neck'], 1367: ['Yonkers', '10710', 'NON-MINORITY', 'WBE'],
            1368: ['Jamaica', 'BLACK', '11432', 'MBE'], 1369: ['NON-MINORITY', 'WBE', '11235', 'Brooklyn'],
            1370: ['BLACK', '10018', 'New York', 'WBE', 'MBE'], 1371: ['11787', 'NON-MINORITY', 'WBE', 'Smithtown'],
            1372: ['7666', 'BLACK', 'Teaneck', 'MBE'], 1373: ['Flushing', 'ASIAN', 'MBE', '11354'],
            1374: ['Jackson Heights', '11370', 'ASIAN', 'MBE'], 1375: ['NON-MINORITY', 'West Nyack', 'WBE', '10994'],
            1376: ['NON-MINORITY', '11797', 'WBE', 'Woodbury'], 1377: ['Brooklyn', '11226', 'BLACK', 'MBE'],
            1378: ['NON-MINORITY', 'WBE', 'New York', '10010'], 1379: ['St. Albans', 'BLACK', '11412', 'MBE'],
            1380: ['10462', 'Bronx', 'HISPANIC', 'WBE', 'MBE'], 1381: ['LBE', '11103', 'Astoria'],
            1382: ['10027', 'New York', 'MBE', 'HISPANIC'], 1383: ['NON-MINORITY', 'WBE', '11215', 'Brooklyn'],
            1384: ['Valley Stream', '11581', 'WBE', 'ASIAN', 'MBE'], 1385: ['White Plains', '10601', 'MBE', 'HISPANIC'],
            1386: ['Webster', 'NON-MINORITY', '14580', 'WBE'], 1387: ['7036', 'Linden', 'ASIAN', 'MBE'],
            1388: ['Lodi', '7644', 'ASIAN', 'MBE'], 1389: ['Brooklyn', 'BLACK', 'MBE', '11233'],
            1390: ['10018', 'New York', 'ASIAN', 'MBE'], 1391: ['NON-MINORITY', 'WBE', '10014', 'New York'],
            1392: ['North Bergen', '7047', 'ASIAN', 'MBE'], 1393: ['7407', 'Elmwood Park', 'ASIAN', 'MBE'],
            1394: ['20005', 'NON-MINORITY', 'Washington', 'WBE'], 1395: ['10075', 'WBE', 'New York'],
            1396: ['BLACK', 'St. Albans', '11412', 'WBE', 'MBE'], 1397: ['NON-MINORITY', 'WBE', '11530', 'Garden City'],
            1398: ['LBE', 'Bronx', 'NON-MINORITY', '10461', 'WBE'], 1399: ['Bronx', 'BLACK', '10469', 'WBE', 'MBE'],
            1400: ['NON-MINORITY', 'WBE', '11201', 'Brooklyn'], 1401: ['10003', 'NON-MINORITY', 'WBE', 'New York'],
            1402: ['11411', 'Cambria Heights', 'ASIAN', 'MBE'], 1403: ['Fort Lee', '7024', 'ASIAN', 'MBE'],
            1404: ['Fort Lee', '7024', 'ASIAN', 'MBE'], 1405: ['10001', 'New York', 'BLACK', 'MBE'],
            1406: ['NON-MINORITY', 'WBE', '10025', 'New York'], 1407: ['East Meadow', '11554', 'WBE', 'ASIAN', 'MBE'],
            1408: ['Brooklyn', 'BLACK', '11208', 'WBE', 'MBE'],
            1409: ['7717', 'NON-MINORITY', 'WBE', 'Avon by the Sea'],
            1410: ['LBE', '11417', 'Ozone Park', 'ASIAN', 'MBE'], 1411: ['NON-MINORITY', 'WBE', 'New York', '10010'],
            1412: ['7666', 'NON-MINORITY', 'WBE', 'Teaneck'], 1413: ['10456', 'Bronx', 'BLACK', 'WBE', 'MBE'],
            1414: ['Paterson', 'BLACK', 'MBE', '7514'], 1415: ['NON-MINORITY', '10023', 'WBE', 'New York'],
            1416: ['11580', 'ASIAN', 'MBE', 'Valley Stream'], 1417: ['Brooklyn', 'BLACK', '11214', 'MBE'],
            1418: ['LBE', '10016', 'New York'], 1419: ['10002', 'New York', 'ASIAN', 'MBE']}
    frequent = apriori_generate_frequent_itemsets(dict, parameters.SUPPORT)
    a_rules = generate_association_rules(frequent, parameters.CONFIDENCE)
    create_tmp_support_table(frequent)
    create_tmp_rule_table(a_rules)
