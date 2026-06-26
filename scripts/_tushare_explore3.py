#!/usr/bin/env python3
"""Explore Tushare API schema and routes."""
import urllib.request
import json

API_BASE = 'http://8.148.76.181:8686/'

def get_json(url):
    req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req, timeout=10)
    return json.loads(resp.read().decode())

# Read openapi.json schemas
print("=== TushareRequest schema ===")
oapi = get_json(API_BASE + 'openapi.json')
schemas = oapi.get('components', {}).get('schemas', {})
ts_req = schemas.get('TushareRequest', {})
print(json.dumps(ts_req, indent=2))

print("\n=== All schema names ===")
print(list(schemas.keys()))

# Read all API paths more carefully
print("\n=== All paths ===")
paths = oapi.get('paths', {})
for path, methods in paths.items():
    print(f"  {path}: {list(methods.keys())}")
    for method, info in methods.items():
        summary = info.get('summary', '')
        params = info.get('parameters', [])
        print(f"    {method}: {summary} (params: {len(params)})")

# Try to get API methods / support
print("\n=== GET /maxad/api/cache/param-superset-methods ===")
try:
    r = get_json(API_BASE + 'maxad/api/cache/param-superset-methods')
    print(json.dumps(r, indent=2)[:2000])
except Exception as e:
    print(f"Failed: {e}")

# Check /maxad/api/stats
print("\n=== GET /maxad/api/stats ===")
try:
    r = get_json(API_BASE + 'maxad/api/stats')
    print(json.dumps(r, indent=2)[:2000])
except Exception as e:
    print(f"Failed: {e}")
