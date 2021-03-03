import random


def generate_dataset(transaction_number=1000000, items_number=100000):
    items = []
    transactions = dict()
    for i in range(items_number):
        items.append("test" + str(i))
    for i in range(transaction_number):
        transactions[i] = []
        for j in range(random.randint(5, 50)):
            transactions[i].append(items[random.randint(0, items_number - 1)])

    print("Generation finished")
    return transactions


# random_large_dataset = generate_dataset()
# my_df = pd.DataFrame(random_large_dataset.values())
# my_df.to_csv('large_csv.csv', index=False, header=False)
