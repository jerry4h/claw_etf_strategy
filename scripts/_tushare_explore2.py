#!/usr/bin/env python3
"""Explore Tushare API auth and routes."""
import urllib.request
import json

API_BASE = 'http://8.148.76.181:8686/'

def post_json(url, data):
    req = urllib.request.Request(url, 
        data=json.dumps(data).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {'http_error': e.code, 'body': e.read().decode()[:500]}
    except Exception as e:
        return {'error': str(e)}

def get_url(url):
    req = urllib.request.Request(url)
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        return {'status': resp.status, 'body': resp.read().decode()[:1000]}
    except urllib.error.HTTPError as e:
        return {'http_error': e.code, 'body': e.read().decode()[:500]}
    except Exception as e:
        return {'error': str(e)}

# Read openapi.json for auth requirements
print("=== openapi.json security schemes ===")
oapi = json.loads(urllib.request.urlopen(API_BASE + 'openapi.json').read())
components = oapi.get('components', {})
security = components.get('securitySchemes', {})
print(json.dumps(security, indent=2))

# The /api POST route details
paths = oapi.get('paths', {})
api_post = paths.get('/api', {}).get('post', {})
print(f"\n=== /api POST ===")
print(f"Summary: {api_post.get('summary', 'N/A')}")
print(f"RequestBody: {json.dumps(api_post.get('requestBody', {}), indent=2)[:500]}")
print(f"Parameters: {json.dumps(api_post.get('parameters', []), indent=2)[:500]}")

# Try with Bearer token header
print("\n=== POST /api with Bearer token ===")
for token_val in ['', 'test', 'demo']:
    data = {'api_name': 'fund_basic', 'token': token_val, 'params': {'market': 'E'}, 'fields': 'ts_code,name'}
    req = urllib.request.Request(API_BASE + 'api',
        data=json.dumps(data).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {token_val}'})
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        print(f"  token='{token_val}': {resp.status} — {resp.read().decode()[:300]}")
    except urllib.error.HTTPError as e:
        print(f"  token='{token_val}': {e.code} — {e.read().decode()[:200]}")

# Try the /{method} path pattern
print("\n=== GET /fund_basic ===")
r = get_url(API_BASE + 'fund_basic')
print(json.dumps(r, indent=2)[:300])

print("\n=== GET /api/fund_basic ===")
r = get_url(API_BASE + 'api/fund_basic')
print(json.dumps(r, indent=2)[:300])

# Try query params approach
print("\n=== POST /fund_basic ===")
r = post_json(API_BASE + 'fund_basic', {'market': 'E'})
print(json.dumps(r, indent=2)[:300])
