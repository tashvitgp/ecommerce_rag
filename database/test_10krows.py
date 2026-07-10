
# Run this in a quick script or notebook
import pandas as pd
from database.connection import engine

df = pd.read_sql("""
    SELECT r.review_id, p.category, r.sentiment
    FROM reviews r
    JOIN products p ON r.product_id = p.product_id
    WHERE r.primary_aspect IS NOT NULL
""", engine)

print(df['category'].value_counts())
print(df['sentiment'].value_counts())