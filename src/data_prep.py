import duckdb

def load_data(csv_path="data/raw/telco_churn.csv", sql_path="sql/features.sql"):
    con = duckdb.connect()
    con.execute(f"""
        CREATE TABLE telco AS
        SELECT * FROM read_csv_auto('{csv_path}')
    """)
    query = open(sql_path).read()
    df = con.execute(query).fetchdf()
    con.close()
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.shape, round(df["Churn"].mean(), 3))
    print(df.head())