from mapreduce_apriori.sep.apriori import load_data

dataset = load_data('million_data.csv')

with open('million_data_postgres.csv', mode='w') as csv_file:
    for transaction, items in dataset.items():
        for item in items:
            csv_file.write(str(transaction) + "," + str(item) + "\n")

