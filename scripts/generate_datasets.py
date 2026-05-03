import os
import random
import pandas as pd
from datetime import datetime, timedelta

def generate_platforms(n=3):
    return pd.DataFrame({
        "platform_id": [1, 2, 3],
        "platform_name": ["Amazon", "BestBuy", "Walmart"]
    })

def generate_customers(n=100):
    customer_ids = [f"C{str(i).zfill(4)}" for i in range(1, n+1)]
    account_created = [
        (datetime.now() - timedelta(days=random.randint(10, 1000))).strftime("%Y-%m-%d %H:%M:%S")
        for _ in range(n)
    ]
    return pd.DataFrame({
        "customer_id": customer_ids,
        "account_created": account_created
    })

def generate_products(n=30):
    product_ids = [f"P{str(i).zfill(4)}" for i in range(1, n+1)]
    categories = ["Electronics", "Books", "Home", "Toys", "Health"]
    brands = ["BrandA", "BrandB", "BrandC", "Generic"]
    return pd.DataFrame({
        "product_id": product_ids,
        "name": [f"Product {i}" for i in range(1, n+1)],
        "category": random.choices(categories, k=n),
        "brand": random.choices(brands, k=n)
    })

def generate_orders(n=200, customers_df=None, products_df=None):
    order_ids = [f"O{str(i).zfill(4)}" for i in range(1, n+1)]
    c_ids = random.choices(customers_df["customer_id"].tolist(), k=n)
    p_ids = random.choices(products_df["product_id"].tolist(), k=n)
    order_timestamp = [
        (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d %H:%M:%S")
        for _ in range(n)
    ]
    return pd.DataFrame({
        "order_id": order_ids,
        "customer_id": c_ids,
        "product_id": p_ids,
        "order_timestamp": order_timestamp
    })

def generate_reviews(n=200, orders_df=None, platforms_df=None):
    review_ids = [f"R{str(i).zfill(4)}" for i in range(1, n+1)]
    
    selected_orders = orders_df.sample(n=n, replace=True).reset_index(drop=True)
    
    ratings = []
    review_texts = []
    
    fraud_templates = [
        "AMAZING!!! BEST PRODUCT EVER!!! BUY THIS NOW!!!",
        "PERFECT PRODUCT!!! HIGHLY RECOMMEND!!!",
        "worth every penny, life changing!",
        "must buy must buy must buy!!!"
    ]
    
    legit_templates = [
        "It's an okay product. Does the job.",
        "Good quality, arrived on time.",
        "Not bad, but could be better.",
        "I really like this, fits well.",
        "Terrible experience, broke after one use."
    ]
    
    for _ in range(n):
        if random.random() < 0.2:
            # Fraud-like
            ratings.append(random.choice([4, 5]))
            review_texts.append(random.choice(fraud_templates))
        else:
            # Legit-like
            ratings.append(random.choice([1, 2, 3, 4, 5]))
            review_texts.append(random.choice(legit_templates))
            
    timestamps = []
    for idx, row in selected_orders.iterrows():
        base_time = datetime.strptime(row["order_timestamp"], "%Y-%m-%d %H:%M:%S")
        review_time = base_time + timedelta(days=random.randint(1, 10))
        timestamps.append(review_time.strftime("%Y-%m-%d %H:%M:%S"))

    verified = random.choices([0, 1], weights=[20, 80], k=n)
    refunded = random.choices([0, 1], weights=[95, 5], k=n)
    p_ids = random.choices(platforms_df["platform_id"].tolist(), k=n)
    
    return pd.DataFrame({
        "review_id": review_ids,
        "customer_id": selected_orders["customer_id"],
        "product_id": selected_orders["product_id"],
        "rating": ratings,
        "review_text": review_texts,
        "review_timestamp": timestamps,
        "verified_purchase": verified,
        "refunded_product": refunded,
        "platform_id": p_ids
    })

def main():
    os.makedirs("datasets", exist_ok=True)
    
    platforms_df = generate_platforms()
    platforms_df.to_csv("datasets/platforms.csv", index=False)
    
    customers_df = generate_customers(100)
    customers_df.to_csv("datasets/customers.csv", index=False)
    
    products_df = generate_products(30)
    products_df.to_csv("datasets/products.csv", index=False)
    
    orders_df = generate_orders(300, customers_df, products_df)
    orders_df.to_csv("datasets/orders.csv", index=False)
    
    reviews_df = generate_reviews(400, orders_df, platforms_df)
    reviews_df.to_csv("datasets/reviews.csv", index=False)
    
    print("Successfully generated datasets in the 'datasets' folder.")

if __name__ == "__main__":
    main()
