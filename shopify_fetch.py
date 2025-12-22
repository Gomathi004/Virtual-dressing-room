# shopify_fetch.py
import requests

SHOPIFY_STORE_URL = "https://virtualdressingroom.myshopify.com/api/2025-10/graphql.json"
STOREFRONT_TOKEN = "c86b17e92460296b5450bb4531762b96"


def _fetch_by_tag(tag: str):
    query = f"""
    {{
      products(first: 20, query: "tag:{tag}") {{
        edges {{
          node {{
            title
            images(first: 14) {{
              edges {{
                node {{
                  url
                }}
              }}
            }}
          }}
        }}
      }}
    }}
    """
    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Storefront-Access-Token": STOREFRONT_TOKEN,
    }

    response = requests.post(
        SHOPIFY_STORE_URL,
        headers=headers,
        json={"query": query},
        timeout=15,
    )
    data = response.json()
    print("Shopify response:", data)

    if "errors" in data or "data" not in data or not data["data"].get("products"):
        return []

    clothes = []
    try:
        for product in data["data"]["products"]["edges"]:
            for img in product["node"]["images"]["edges"]:
                url = img["node"].get("url")
                if url:
                    clothes.append(url)
    except Exception as e:
        print("Parsing error:", e)
        return []
    return clothes


def get_clothes(gender: str = "men", kind: str = "top"):
    """
    kind: 'top' or 'bottom'
    gender: 'men' or 'women'
    """
    if kind not in ("top", "bottom"):
        kind = "top"
    if gender not in ("men", "women"):
        gender = "men"

    tag = f"{gender}-{kind}"  # e.g. men-top, women-bottom
    return _fetch_by_tag(tag)
