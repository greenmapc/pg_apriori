def read_file_data(file_name):
    with open(file_name) as file_iter:
        for line in file_iter:
            line = line.strip().rstrip(",")
            record = frozenset(line.split(","))
            yield record


def get_item_set_transaction_list(data_iterator):
    transaction_list = list()
    for record in data_iterator:
        transaction = frozenset(record)
        transaction_list.append(transaction)
    return transaction_list


file_data = read_file_data("../INTEGRATED-DATASET.csv")
transactions = get_item_set_transaction_list(file_data)

data_for_csv = []
for i in range(len(transactions)):
    for element in transactions[i]:
        data_for_csv.append({'who': i, 'what': element})
print('a')

import csv

with open('data.csv', 'w', newline='') as file:
    fieldnames = ['who', 'what']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for item in data_for_csv:
        writer.writerow(item)