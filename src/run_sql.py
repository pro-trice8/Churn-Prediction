import duckdb

con = duckdb.connect()
con.execute("""
    CREATE TABLE telco AS
    SELECT * FROM read_csv_auto('data/raw/telco_churn.csv')
""")
query = open("sql/explore.sql").read()
print(con.execute(query).fetchdf())