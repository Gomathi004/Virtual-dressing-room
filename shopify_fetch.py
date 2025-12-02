# shopify_fetch.py
import requests

SHOPIFY_STORE_URL = "https://virtualdressingroom.myshopify.com/api/2025-10/graphql.json"

# Storefront API access token (NOT Admin token)
STOREFRONT_TOKEN = "c86b17e92460296b5450bb4531762b96"  # replace

def get_clothes():
    query = """
    {
      products(first: 14) {
        edges {
          node {
            title
            images(first: 14) {
              edges {
                node {
                  url
                }
              }
            }
          }
        }
      }
    }
    """

    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Storefront-Access-Token": STOREFRONT_TOKEN,
    }

    response = requests.post(
        SHOPIFY_STORE_URL,
        headers=headers,
        json={"query": query},
    )

    data = response.json()

    if "data" not in data or "products" not in data["data"]:
        print("Error: Shopify returned no product data.")
        print(data)
        return []

    clothes = []
    try:
        for product in data["data"]["products"]["edges"]:
            for img in product["node"]["images"]["edges"]:
                clothes.append(img["node"]["url"])
    except Exception as e:
        print("Parsing error:", e)
        return []

    return clothes