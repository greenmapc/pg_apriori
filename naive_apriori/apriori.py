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


def getItemSetTransactionList(dict):
    transactionList = list()
    itemSet = set()
    for record in dict.values():
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))  # Generate 1-itemSets
    return itemSet, transactionList


def runApriori(data, minSupport, minConfidence):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - rules ((pretuple, posttuple), confidence)
    """
    itemSet, transactionList = getItemSetTransactionList(data)

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

def naive_apriori_run(data, support, confidence):
    runApriori(data, support, confidence)
