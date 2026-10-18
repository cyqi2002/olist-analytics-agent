"""Load the Olist CSVs into a local DuckDB file with clean table names and types.

Usage:
    python data/load.py                      # expects CSVs in data/raw/
    python data/load.py --raw path/to/csvs --db data/olist.duckdb

Get the CSVs from https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
(Download -> unzip into data/raw/).
"""

import argparse
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent

# table name -> (csv file, explicit column types for columns auto-detection gets wrong)
TABLES = {
    "customers": ("olist_customers_dataset.csv", {"customer_zip_code_prefix": "VARCHAR"}),
    "sellers": ("olist_sellers_dataset.csv", {"seller_zip_code_prefix": "VARCHAR"}),
    "geolocation": ("olist_geolocation_dataset.csv", {"geolocation_zip_code_prefix": "VARCHAR"}),
    "orders": (
        "olist_orders_dataset.csv",
        {
            "order_purchase_timestamp": "TIMESTAMP",
            "order_approved_at": "TIMESTAMP",
            "order_delivered_carrier_date": "TIMESTAMP",
            "order_delivered_customer_date": "TIMESTAMP",
            "order_estimated_delivery_date": "TIMESTAMP",
        },
    ),
    "order_items": (
        "olist_order_items_dataset.csv",
        {"shipping_limit_date": "TIMESTAMP", "price": "DECIMAL(10,2)", "freight_value": "DECIMAL(10,2)"},
    ),
    "order_payments": ("olist_order_payments_dataset.csv", {"payment_value": "DECIMAL(10,2)"}),
    "order_reviews": (
        "olist_order_reviews_dataset.csv",
        {"review_creation_date": "TIMESTAMP", "review_answer_timestamp": "TIMESTAMP"},
    ),
    "products": ("olist_products_dataset.csv", {}),
    "category_translation": ("product_category_name_translation.csv", {}),
}

# the source data misspells "length"; fix it so the model doesn't have to remember the typo
RENAMES = {
    "products": {
        "product_name_lenght": "product_name_length",
        "product_description_lenght": "product_description_length",
    }
}


def load(raw_dir: Path, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    for table, (filename, types) in TABLES.items():
        csv = raw_dir / filename
        if not csv.exists():
            raise FileNotFoundError(f"{csv} not found - download the Olist dataset into {raw_dir}")
        options = "header=true"
        if types:
            options += ", types={" + ", ".join(f"'{c}': '{t}'" for c, t in types.items()) + "}"
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM read_csv(?, {options})", [str(csv)])
        for old, new in RENAMES.get(table, {}).items():
            con.execute(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")
        n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"{table:<22} {n:>8,} rows")
    con.close()
    print(f"\nwrote {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "raw")
    parser.add_argument("--db", type=Path, default=ROOT / "olist.duckdb")
    args = parser.parse_args()
    load(args.raw, args.db)
