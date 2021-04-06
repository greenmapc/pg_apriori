from map_reduce_trie_extension import script
from map_reduce_trie_extension.script import run
from mapreduce_apriori.test import parameters
from naive_apriori.apriori import naive_apriori_run

simple_dataset = {0: ['LBE', '11204', 'Brooklyn'], 1: ['BLACK', 'Cambria Heights', '11411', 'WBE', 'MBE'],
                  2: ['Yorktown Heights', '10598', 'BLACK', 'MBE'], 3: ['11561', 'BLACK', 'MBE', 'Long Beach'],
                  4: ['11235', 'Brooklyn', 'ASIAN', 'MBE'], 5: ['New York', '10010', 'WBE', 'ASIAN', 'MBE'],
                  6: ['10026', 'New York', 'ASIAN', 'MBE'], 7: ['New York', 'BLACK', '10026', 'MBE'],
                  8: ['10034', 'New York', 'MBE', 'HISPANIC'], 9: ['BLACK', '10303', 'Staten Island', 'WBE', 'MBE'],
                  10: ['10018', 'New York', 'ASIAN', 'MBE'], 11: ['New York', 'HISPANIC', '10034', 'WBE', 'MBE'],
                  12: ['New York', 'WBE', 'ASIAN', 'MBE', '10013'], 13: ['Jamaica', 'BLACK', 'MBE', '11434'],
                  14: ['NON-MINORITY', 'WBE', 'New York', '10022'], 15: ['10304', 'BLACK', 'MBE', 'Staten Island'],
                  16: ['Bronx', 'BLACK', '10454', 'MBE'], 17: ['New Rochelle', 'NON-MINORITY', 'WBE', '10801'],
                  18: ['10301', 'NON-MINORITY', 'WBE', 'Staten Island'],
                  19: ['10006', 'NON-MINORITY', 'WBE', 'New York'],
                  20: ['Brooklyn', 'BLACK', '11239', 'MBE'], 21: ['7035', 'Lincoln Park', 'MBE', 'HISPANIC'],
                  22: ['BLACK', 'New York', '10027', 'WBE', 'MBE'],
                  23: ['10310', 'NON-MINORITY', 'WBE', 'Staten Island'],
                  24: ['New York', 'ASIAN', 'MBE', '10013'], 25: ['NON-MINORITY', 'Cliffside Park', 'WBE', '7010'],
                  26: ['10456', 'Bronx', 'BLACK', 'WBE', 'MBE'], 27: ['LBE', '10003', 'New York'],
                  28: ['10303', 'Staten Island', 'MBE', 'HISPANIC'], 29: ['10001', 'New York', 'ASIAN', 'MBE'],
                  30: ['New York', '11435', 'BLACK', 'MBE'], 31: ['Ozone Park', 'WBE', '11417'],
                  32: ['Lawrence', '11559', 'NON-MINORITY', 'WBE'], 33: ['LBE', 'Brooklyn', '11230', 'ASIAN', 'MBE'],
                  34: ['11563', 'Lynbrook', 'MBE', 'HISPANIC'], 35: ['Newark', 'BLACK', 'MBE', '7104'],
                  36: ['11356', 'NON-MINORITY', 'WBE', 'College Point'],
                  37: ['Berkeley Heights', '7922', 'ASIAN', 'MBE'],
                  38: ['LBE', 'New York', 'HISPANIC', '10040', 'WBE', 'MBE'],
                  39: ['East Elmhurst', '11370', 'ASIAN', 'MBE'],
                  40: ['LBE', 'Astoria', '11106'], 41: ['MBE', 'New York', 'HISPANIC', 'WBE', '10001'],
                  42: ['LBE', 'Bronx', 'BLACK', '10457', 'MBE'],
                  43: ['South Ozone Park', '11420', 'BLACK', 'WBE', 'MBE'],
                  44: ['10920', 'Congers', 'ASIAN', 'MBE'], 45: ['Bronx', '10456', 'BLACK', 'MBE'],
                  46: ['11219', 'Brooklyn', 'ASIAN', 'MBE'], 47: ['11360', 'ASIAN', 'MBE', 'Bayside'],
                  48: ['10001', 'NON-MINORITY', 'WBE', 'New York'], 49: ['10462', 'Bronx', 'MBE', 'HISPANIC'],
                  50: ['LBE', 'Bronx', 'BLACK', '10470', 'MBE'], 51: ['11803', 'Plainview', 'ASIAN', 'MBE']}


def main_test():
    expected_result = [(['ASIAN'], 14), (['BLACK'], 17), (['Bronx'], 6), (['HISPANIC'], 8), (['LBE'], 7),
                       (['MBE'], 39), (['NON-MINORITY'], 9), (['New York'], 17), (['WBE'], 20),
                       ([('ASIAN', 'MBE')], 14), ([('ASIAN', 'New York')], 6), ([('BLACK', 'MBE')], 17),
                       ([('Bronx', 'MBE')], 6), ([('HISPANIC', 'MBE')], 8), ([('MBE', 'New York')], 13),
                       ([('MBE', 'WBE')], 10), ([('NON-MINORITY', 'WBE')], 9), ([('New York', 'WBE')], 9),
                       ([('ASIAN', 'MBE', 'New York')], 6), ([('MBE', 'New York', 'WBE')], 6)]
    # result = run_algorithm(simple_dataset)
    result = run(simple_dataset, 0, 0)
    print(expected_result)
    print("expected_result")
    print("result")
    print(result)
    if (result == expected_result):
        print("OK")
    else:
        print("NOT OK")


def apriori_for_postgres_code_test():
    expected_result = [(['ASIAN'], 14), (['BLACK'], 17), (['Bronx'], 6), (['HISPANIC'], 8), (['LBE'], 7),
                       (['MBE'], 39), (['NON-MINORITY'], 9), (['New York'], 17), (['WBE'], 20),
                       ([('ASIAN', 'MBE')], 14), ([('ASIAN', 'New York')], 6), ([('BLACK', 'MBE')], 17),
                       ([('Bronx', 'MBE')], 6), ([('HISPANIC', 'MBE')], 8), ([('MBE', 'New York')], 13),
                       ([('MBE', 'WBE')], 10), ([('NON-MINORITY', 'WBE')], 9), ([('New York', 'WBE')], 9),
                       ([('ASIAN', 'MBE', 'New York')], 6), ([('MBE', 'New York', 'WBE')], 6)]
    frequent_items, rules = run(simple_dataset, parameters.SUPPORT, parameters.CONFIDENCE)
    print(expected_result)
    print("expected_result")
    print("result")
    naive_apriori_result = naive_apriori_run(simple_dataset, parameters.SUPPORT, parameters.CONFIDENCE)
    print(naive_apriori_result)
    print(frequent_items)
    if frequent_items == expected_result:
        print("OK")
    else:
        print("NOT OK")


main_test()
# apriori_for_postgres_code_test()
