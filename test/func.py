def abacaba():
    return 5


def test(name):
    return name + abacaba()


from datetime import datetime

dt_string = datetime.now().strftime("%Y%m%d%H%M%S")
result_table_name = "pg_apriori_result_" + dt_string

print(result_table_name)


def printResults(items, rules):
    ans = "\n------------------------ ITEM SETS WITH SUPPORT:"
    for item, support in sorted(items, key=lambda x: x[1]):
        ans += "item: %s , %.3f" % (str(item), support)
        ans += "\n"
    ans += "\n------------------------ RULES:"
    for rule, confidence in sorted(rules, key=lambda x: x[1]):
        pre, post = rule
        ans += "rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence)
        ans += "\n"
    return ans

create_table_query = "CREATE TABLE " + result_table_name +\
                     "(" + \
                     "items VARCHAR []," + \
                     "support double precision" +\
                     ")"

insert_table_query = "INSERT INTO " + result_table_name +\
                    "(items, support)" + \
                     " VALUES (ARRAY[%s], %1.3f)"
print(insert_table_query % ("\'asd\', \'A\'", 4.3))
print(create_table_query)