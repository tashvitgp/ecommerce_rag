import pandas as pd
import uuid
from database.connection import engine
from database.schemas import Base

df = pd.read_csv("data/flipkart_reviews.csv")

# normalize/clean columns based on actual CSV headers
df['Review'] = df['Review'].fillna('')
df['Summary'] = df['Summary'].fillna('')
df['review_text'] = df['Summary'] + ". " + df['Review']

# CSV uses lowercase 'product_name' and 'product_price'
unique_products = df['product_name'].unique()
product_id_mapping = {name: str(uuid.uuid4()) for name in unique_products}
df['product_id'] = df['product_name'].map(product_id_mapping)

df['review_id'] = [str(uuid.uuid4()) for _ in range(len(df))]

df['brand'] = df['product_name'].apply(lambda x: str(x).split()[0] if pd.notnull(x) else "Unknown")
# keep a consistent column name for downstream code
df['product_name'] = df['product_name']
df['price'] = pd.to_numeric(df['product_price'].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
df['rating'] = pd.to_numeric(df['Rate'], errors='coerce')
df['category'] = "General"
df['timestamp'] = pd.Timestamp.now()

products = df[['product_id', 'product_name', 'brand', 'price', 'category']].drop_duplicates(subset=['product_id'])
reviews_raw = df[['review_id', 'product_id', 'review_text', 'rating', 'timestamp']]

Base.metadata.create_all(bind=engine)

products.to_sql("products", engine, if_exists="append", index=False, method="multi", chunksize=2000)
reviews_raw.to_sql("reviews", engine, if_exists="append", index=False, method="multi", chunksize=2000)