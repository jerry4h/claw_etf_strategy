#!/usr/bin/env python3
"""Explore Tushare API proxy endpoints."""
import urllib.request
import json

API_BASE = 'http://8.148.76.181:8686/'

def get_json(url):
    req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req, timeout=10)
    return json.loads(resp.read().decode())

def post_json(url, data):
    req = urllib.request.Request(url, 
        data=json.dumps(data).encode(),
        headers={'Content-Type': 'application/json'})
    resp = urllib.request.urlopen(req, timeout=15)
    return json.loads(resp.read().decode())

# Try /docs endpoint
print("=== /docs ===")
try:
    docs = get_json(API_BASE + 'docs')
    print(json.dumps(docs, indent=2)[:2000])
except Exception as e:
    print(f"Failed: {e}")

# Try /openapi.json
print("\n=== /openapi.json ===")
try:
    oapi = get_json(API_BASE + 'openapi.json')
    paths = oapi.get('paths', {})
    print(f"Paths: {list(paths.keys())[:20]}")
    if '/api' in paths:
        print(f"/api methods: {list(paths['/api'].keys())}")
except Exception as e:
    print(f"Failed: {e}")

# Try POST to /api with fund_basic
print("\n=== POST /api (fund_basic) ===")
try:
    # Tushare standard format
    result = post_json(API_BASE + 'api', {
        'api_name': 'fund_basic',
        'token': '',  # proxy might not need token
        'params': {'market': 'E'},  # E = ETF
        'fields': 'ts_code,name,management,found_date,benchmark,invest_type'
    })
    print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")

# Also try GET on /api/fund_basic
print("\n=== GET /api/fund_basic ===")
try:
    result = get_json(API_BASE + 'api/fund_basic')
    print(json.dumps(result, indent=2, ensure_ascii=False)[:1000])
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")
