# Read data from csv file
from naive_apriori.apriori import run_apriori, printResults


def read_file_data(file_name):
    with open(file_name) as file_iter:
        for line in file_iter:
            line = line.strip().rstrip(",")  # Remove trailing comma
            record = frozenset(line.split(","))
            yield record


def main():
    file_name = "../simple_test.csv"
    min_support = 0.17
    min_confidence = 0.68

    file_data = read_file_data(file_name)

    items, rules = run_apriori(file_data, min_support, min_confidence)

    printResults(items, rules)


main()
