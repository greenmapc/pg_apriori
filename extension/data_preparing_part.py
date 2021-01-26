import json


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


example = '{ "table_name":"table", "transaction_column":"who", "item_column":"what"}'
data = prepare_data_from_json(example)
print("A")
