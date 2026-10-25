"""
Seeds demo data by calling your own running API -- works against localhost
or your deployed Render URL. This is deliberately done over HTTP rather
than raw SQL: it guarantees passwords are hashed correctly and knowledge
base chunks get real embeddings, using the exact same code paths a real
user would.

Usage:
    BASE_URL=http://localhost:8000 python scripts/seed_demo_data.py
    BASE_URL=https://your-app.onrender.com python scripts/seed_demo_data.py
"""
import os

import httpx

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
ADMIN_EMAIL = "demo@shop.com"
ADMIN_PASSWORD = "DemoPass123!"
CUSTOMER_EMAIL = "customer@example.com"

PRODUCTS = [
    {"sku": "HP-100", "name": "Wireless Headphones", "description": "Over-ear, noise cancelling, 30hr battery.", "category": "Electronics", "price": 89.99, "stock_quantity": 25},
    {"sku": "SH-200", "name": "Running Shoes", "description": "Lightweight trail running shoes, breathable mesh.", "category": "Footwear", "price": 64.50, "stock_quantity": 40},
    {"sku": "BT-300", "name": "Insulated Water Bottle", "description": "Keeps drinks cold for 24 hours, 750ml.", "category": "Accessories", "price": 19.99, "stock_quantity": 100},
    {"sku": "BP-400", "name": "Laptop Backpack", "description": "Water-resistant, fits up to 16-inch laptops.", "category": "Bags", "price": 45.00, "stock_quantity": 15},
    {"sku": "WT-500", "name": "Smart Watch", "description": "Heart-rate tracking, 7-day battery, GPS.", "category": "Electronics", "price": 129.00, "stock_quantity": 8},
]

DOCUMENTS = [
    ("Return Policy", "Items can be returned within 30 days of delivery for a full refund, as long as they're unused and in original packaging. Return shipping is free for defective items."),
    ("Shipping Policy", "Standard shipping takes 3-5 business days and is free on orders over $50. Express shipping (1-2 days) is available at checkout for an extra fee."),
    ("Warranty Info", "All electronics come with a 1-year manufacturer warranty covering defects. Warranty does not cover accidental damage or normal wear and tear."),
]


def main() -> None:
    client = httpx.Client(base_url=BASE_URL, timeout=30)

    # 1. Register the demo admin (auto-creates their store). Fine if it
    #    already exists -- we just log in instead.
    r = client.post("/auth/register", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    if r.status_code not in (201, 400):
        r.raise_for_status()

    r = client.post(
        "/auth/jwt/login",
        data={"username": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    r.raise_for_status()
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    store = client.get("/stores/me", headers=headers).json()
    print(f"Store public_api_key: {store['public_api_key']}")

    # 2. Products
    product_ids = []
    for p in PRODUCTS:
        r = client.post("/products", json=p, headers=headers)
        if r.status_code == 201:
            product_ids.append(r.json()["id"])
        elif r.status_code == 400:
            print(f"  (skipping {p['sku']}, already exists)")
        else:
            r.raise_for_status()
    print(f"Products created: {len(product_ids)}")

    # 3. Two orders under one customer email: one shipped (order tracking
    #    demo) and one delivered (return demo).
    bysku = {p["sku"]: p["id"] for p in client.get("/products", headers=headers).json()}
    for sku, qty, status in (("HP-100", 1, "shipped"), ("SH-200", 2, "delivered")):
        if sku not in bysku:
            continue
        r = client.post(
            "/orders",
            json={
                "customer_email": CUSTOMER_EMAIL,
                "customer_name": "Demo Customer",
                "items": [{"product_id": bysku[sku], "quantity": qty}],
            },
            headers=headers,
        )
        r.raise_for_status()
        oid = r.json()["id"]
        client.patch(
            f"/orders/{oid}", json={"status": status, "tracking_number": f"TRK-{oid}-DEMO"}, headers=headers
        ).raise_for_status()
        print(f"Order created: id={oid} status={status}")

    # 4. Knowledge base: policy docs + the product catalog itself
    for title, content in DOCUMENTS:
        client.post("/knowledge-base/documents", json={"title": title, "content": content}, headers=headers)
    client.post("/knowledge-base/sync-products", headers=headers)
    print("Knowledge base seeded (policies + product catalog).")

    print("\n--- Demo credentials ---")
    print(f"Admin login: {ADMIN_EMAIL} / {ADMIN_PASSWORD}")
    print(f"Store key (X-Store-Key header for /chat): {store['public_api_key']}")
    print(f"Test customer email for order lookups: {CUSTOMER_EMAIL}")


if __name__ == "__main__":
    main()
