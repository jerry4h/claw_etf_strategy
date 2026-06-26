#!/usr/bin/env python3
"""Check Tushare admin page for token registration."""
import urllib.request
import json

API_BASE = 'http://8.148.76.181:8686/'

# Check openapi.json for /buyers endpoints - maybe we can register
req = urllib.request.Request(API_BASE + 'openapi.json')
oapi = json.loads(urllib.request.urlopen(req))

# Look for buyer creation endpoint
paths = oapi.get('paths', {})
buyer_post = paths.get('/maxad/api/buyers', {}).get('post', {})
print("=== Buyer Create Endpoint ===")
print(f"Summary: {buyer_post.get('summary', 'N/A')}")
req_body = buyer_post.get('requestBody', {})
print(f"RequestBody schema: {json.dumps(req_body, indent=2)[:1000]}")

# Check schemas for BuyerCreate
schemas = oapi.get('components', {}).get('schemas', {})
print(f"\n=== BuyerCreate Schema ===")
bc = schemas.get('BuyerCreate', {})
print(json.dumps(bc, indent=2))

print(f"\n=== BuyerUpdate Schema ===")
bu = schemas.get('BuyerUpdate', {})
print(json.dumps(bu, indent=2))
