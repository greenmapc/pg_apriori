# dataset = load_data('kaggle_dataset.txt')
# dataset = load_data('million_data.csv')
import timeit

from mapreduce_apriori.apriori import run
from mapreduce_apriori.datasets_generator import generate_dataset
from naive_apriori.apriori import naive_apriori_run

dataset = generate_dataset(100000, 100000)
support = 0
confidence = 0


def naive_algorithm():
    start = timeit.default_timer()
    naive_apriori_run(dataset, support, confidence)
    finish = timeit.default_timer()
    print(start - finish)


def mapreduce_trie_algorithm():
    start = timeit.default_timer()
    run(dataset, support, confidence)
    finish = timeit.default_timer()
    print(start - finish)

mapreduce_trie_algorithm()
