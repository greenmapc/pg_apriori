# pg_apriori 
Extension for generation of association rules

## Extension init 

``
sudo make install
``

## Extension creation
```sql
CREATE EXTENSION pg_apriori;
```

## Data for running of extension
```json
{
  "table_name":"test", 
  "transaction_column":"transaction", 
  "item_column":"item", 
  "min_support": 30, 
  "min_confidence": 50
}
```
`table_name` - name of table for analysis

`transaction_column` - column describes number of transaction

`item_column` - column describes item in transaction

`min_support` - minimal support in percent

`min_confidence` - minimal confidence in percent

## Running of extension

```sql
SELECT * from apriori('{ ' ||
                      '"table_name":"iter1_test_table", ' ||
                      '"transaction_column":"who", ' ||
                      '"item_column":"what", ' ||
                      '"min_support":3, ' ||
                      '"min_confidence":5' ||
                      '}'
    );
```

## Result of extension

| support_table  | rules_table |
| ------------- | ------------- |
| support_table_name  | rules_table_name  |

## Content of tables

1. support_table

```sql
SELECT * FROM support_table
```

| items  | support |
| ------------- | ------------- |
| ...  | ...  |
| {BLACK}  | 51.211  |
| {MBE,WBE}  | 70.07  |
| {NON-MINORITY,New York,WBE}  | 63.38  |
| ...  | ...  |


2. rules_table

```sql
SELECT * FROM support_table
```

| items_from  | items_to | confidence | 
| ------------- | ------------- | ------------- | 
| ...  | ...  | ...  |
| {HISPANIC}  | {New York}  | 30.901  |
| {MBE,New York}  | {MBE,New York}  | 35.124  |
| {Bronx}  | {MBE,BLACK}  | 52.212  |
| {MBE,New York}  | {BLACK}  | 33.884  |
| ...  | ...  | ...  |
