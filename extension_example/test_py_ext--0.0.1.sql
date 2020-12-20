\echo Use "CREATE EXTENSION test_py_ext" to load this file. \quit

CREATE OR REPLACE FUNCTION test_hello(IN f int, IN s int) RETURNS int AS
$$

    return f + s;
$$
LANGUAGE 'plpython3u' VOLATILE;

