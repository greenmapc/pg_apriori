\echo Use "CREATE EXTENSION create_temp_table_ext" to load this file. \quit

CREATE OR REPLACE FUNCTION create_test_temp_table() RETURNS VARCHAR AS
$$
    plpy.execute("DROP TABLE IF EXISTS pg_apriori_result;")

    plpy.execute(
        "CREATE TEMP TABLE pg_apriori_result "
        "("
        "   support double precision, "
        "   items VARCHAR []"
        ");"
    )

    plpy.execute(
        "insert into pg_apriori_result(support, items) "
        "values (0.175, ARRAY ['New York', 'WBE']), "
        "       (0.200, ARRAY ['MBE', 'ASIAN']);"
    )
$$
LANGUAGE 'plpython3u' VOLATILE;
