# from collections import defaultdict
# from itertools import chain, combinations
#
#
# def subsets(arr):
#     """ Returns non empty subsets of arr"""
#     return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])
#
#
# def return_items_with_min_support(item_set, transaction_list, min_support, freq_set):
#     """calculates the support for items in the itemSet and returns a subset
#     of the itemSet each of whose elements satisfies the minimum support"""
#     item_set_with_min_support = set()
#     local_set = defaultdict(int)
#
#     for item in item_set:
#         for transaction in transaction_list:
#             if item.issubset(transaction):
#                 freq_set[item] += 1
#                 local_set[item] += 1
#
#     for item, count in local_set.items():
#         support = float(count) / len(transaction_list)
#
#         if support >= min_support:
#             item_set_with_min_support.add(item)
#
#     return item_set_with_min_support
#
#
# def generate_candidates(item_set, length):
#     """Join a set with itself and returns the n-element itemsets"""
#     return set(
#         [i.union(j) for i in item_set for j in item_set if len(i.union(j)) == length]
#     )
#
#
# # get list of transactions and set of all transactions elements
# def get_item_set_transaction_list(data_iterator):
#     transaction_list = list()
#     item_set = set()
#     for record in data_iterator:
#         transaction = frozenset(record)
#         transaction_list.append(transaction)
#         for item in transaction:
#             item_set.add(frozenset([item]))
#     return item_set, transaction_list
#
#
# def run_apriori(data_iter, min_support, min_confidence):
#     """
#     run the apriori algorithm. data_iter is a record iterator
#     Return both:
#      - items (tuple, support)
#      - rules ((pretuple, posttuple), confidence)
#     """
#     item_set, transaction_list = get_item_set_transaction_list(data_iter)
#
#     # map of item sets with frequency
#     # (key=n-itemSets,value=support)
#     freq_set = defaultdict(int)
#
#     # key = item set length
#     # value = list of item sets
#     large_set = dict()
#
#     # find more frequently item set
#     one_item_satisfied_set = return_items_with_min_support(item_set, transaction_list, min_support, freq_set)
#
#     current_large_set = one_item_satisfied_set
#
#     item_length = 2
#     while current_large_set != set([]):
#         large_set[item_length - 1] = current_large_set
#         current_large_set = generate_candidates(current_large_set, item_length)
#         current_satisfied_set = return_items_with_min_support(
#             current_large_set, transaction_list, min_support, freq_set
#         )
#         current_large_set = current_satisfied_set
#         item_length = item_length + 1
#
#     def get_support(item):
#         """local function which Returns the support of an item"""
#         return float(freq_set[item]) / len(transaction_list)
#
#     # list of tuple (item set, support)
#     items_result = []
#     for key, value in large_set.items():
#         # calculate support for each item set
#         items_result.extend([(tuple(item), get_support(item)) for item in value])
#
#     # list of tuple (item set, confidence)
#     rules_result = []
#     for key, value in list(large_set.items())[1:]:
#         for item in value:
#             # separate set by elements
#             _subsets = map(frozenset, [x for x in subsets(item)])
#             # calculate confidence for each element of item set
#             for element in _subsets:
#                 remain = item.difference(element)
#                 if len(remain) > 0:
#                     confidence = get_support(item) / get_support(element)
#                     if confidence >= min_confidence:
#                         rules_result.append(((tuple(element), tuple(remain)), confidence))
#     return items_result, rules_result
#
#
# # todo change output - transform to table
# def printResults(items, rules):
#     """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
#     print("\n------------------------ ITEM SETS WITH SUPPORT:")
#     for item, support in sorted(items, key=lambda x: x[1]):
#         print("item: %s , %.3f" % (str(item), support))
#     print("\n------------------------ RULES:")
#     for rule, confidence in sorted(rules, key=lambda x: x[1]):
#         pre, post = rule
#         print("rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence))
#
#
# def to_str_results(items, rules):
#     """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
#     i, r = [], []
#     for item, support in sorted(items, key=lambda x: x[1]):
#         x = "item: %s , %.3f" % (str(item), support)
#         i.append(x)
#
#     for rule, confidence in sorted(rules, key=lambda x: x[1]):
#         pre, post = rule
#         x = "Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence)
#         r.append(x)
#
#     return i, r

"""
Description     : Simple Python implementation of the Apriori Algorithm
Usage:
    $python apriori.py -f DATASET.csv -s minSupport  -c minConfidence
    $python apriori.py -f DATASET.csv -s 0.15 -c 0.6
"""

import sys

from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser


def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
    """calculates the support for items in the itemSet and returns a subset
    of the itemSet each of whose elements satisfies the minimum support"""
    _itemSet = set()
    localSet = defaultdict(int)

    for item in itemSet:
        for transaction in transactionList:
            if item.issubset(transaction):
                freqSet[item] += 1
                localSet[item] += 1

    for item, count in localSet.items():
        support = float(count) / len(transactionList)

        if support >= minSupport:
            _itemSet.add(item)

    return _itemSet


def joinSet(itemSet, length):
    """Join a set with itself and returns the n-element itemsets"""
    return set(
        [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length]
    )


def getItemSetTransactionList():
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
            50: ['LBE', 'Bronx', 'BLACK', '10470', 'MBE'], 51: ['11803', 'Plainview', 'ASIAN', 'MBE']}
    transactionList = list()
    itemSet = set()
    for record in dict.values():
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))  # Generate 1-itemSets
    return itemSet, transactionList


def runApriori(data_iter, minSupport, minConfidence):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - rules ((pretuple, posttuple), confidence)
    """
    itemSet, transactionList = getItemSetTransactionList()

    freqSet = defaultdict(int)
    largeSet = dict()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport

    assocRules = dict()
    # Dictionary which stores Association Rules

    oneCSet = returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet)

    currentLSet = oneCSet
    k = 2
    while currentLSet != set([]):
        largeSet[k - 1] = currentLSet
        currentLSet = joinSet(currentLSet, k)
        currentCSet = returnItemsWithMinSupport(
            currentLSet, transactionList, minSupport, freqSet
        )
        currentLSet = currentCSet
        k = k + 1

    def getSupport(item):
        """local function which Returns the support of an item"""
        return float(freqSet[item]) / len(transactionList)

    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item)) for item in value])

    toRetRules = []
    for key, value in list(largeSet.items())[1:]:
        for item in value:
            _subsets = map(frozenset, [x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)
                if len(remain) > 0:
                    confidence = getSupport(item) / getSupport(element)
                    if confidence >= minConfidence:
                        toRetRules.append(((tuple(element), tuple(remain)), confidence))
    return toRetItems, toRetRules


def printResults(items, rules, min_support):
    """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
    print(min_support)
    for item, support in sorted(items, key=lambda x: x[1]):
        if support >= min_support:
            print("item: %s , %.3f" % (str(item), support * 52))
    print("\n------------------------ RULES:")
    for rule, confidence in sorted(rules, key=lambda x: x[1]):
        pre, post = rule
        print("Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence))


def to_str_results(items, rules):
    """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
    i, r = [], []
    for item, support in sorted(items, key=lambda x: x[1]):
        x = "item: %s , %.3f" % (str(item), support)
        i.append(x)

    for rule, confidence in sorted(rules, key=lambda x: x[1]):
        pre, post = rule
        x = "Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence)
        r.append(x)

    return i, r


def dataFromFile(fname):
    """Function which reads from the file and yields a generator"""
    with open(fname, "rU") as file_iter:
        for line in file_iter:
            line = line.strip().rstrip(",")  # Remove trailing comma
            record = frozenset(line.split(","))
            yield record


if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option(
        "-f", "--inputFile", dest="input", help="filename containing csv", default=None
    )
    optparser.add_option(
        "-s",
        "--minSupport",
        dest="minS",
        help="minimum support value",
        default=0.15,
        type="float",
    )
    optparser.add_option(
        "-c",
        "--minConfidence",
        dest="minC",
        help="minimum confidence value",
        default=0.6,
        type="float",
    )

    (options, args) = optparser.parse_args()

    inFile = None
    if options.input is None:
        inFile = sys.stdin
    elif options.input is not None:
        inFile = dataFromFile(options.input)
    else:
        print("No dataset filename specified, system with exit\n")
        sys.exit("System will exit")

    minSupport = options.minS
    minConfidence = options.minC

    items, rules = runApriori(inFile, minSupport, minConfidence)

    printResults(items, rules, minSupport)