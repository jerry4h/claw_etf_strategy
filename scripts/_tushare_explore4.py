#!/usr/bin/env python3
"""Try all Tushare API access methods."""
import urllib.request
import json

API_BASE = 'http://8.148.76.181:8686/'

def post_json(url, data, headers=None):
    if headers is None:
        headers = {}
    headers.setdefault('Content-Type', 'application/json')
    req = urllib.request.Request(url, 
        data=json.dumps(data).encode(),
        headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        return {'status': resp.status, 'body': resp.read().decode()}
    except urllib.error.HTTPError as e:
        return {'http_error': e.code, 'body': e.read().decode()[:500]}
    except Exception as e:
        return {'error': str(e)}

# Try POST to / (root - "Tushare Compatible Post") with no token
print("=== POST / (Tushare Compatible, no token) ===")
r = post_json(API_BASE, {
    'api_name': 'fund_basic',
    'params': {'market': 'E'},
    'fields': 'ts_code,name'
})
print(json.dumps(r, indent=2)[:500])

# Try POST to /{method}
print("\n=== POST /fund_basic ===")
r = post_json(API_BASE + 'fund_basic', {
    'api_name': 'fund_basic',
    'params': {'market': 'E'},
    'fields': 'ts_code,name'
})
print(json.dumps(r, indent=2)[:500])

# Try //{method}
print("\n=== POST //fund_basic ===")
r = post_json(API_BASE.rstrip('/') + '//fund_basic', {
    'api_name': 'fund_basic',
    'params': {'market': 'E'},
    'fields': 'ts_code,name'
})
print(json.dumps(r, indent=2)[:500])

# The schema says token is optional (anyOf: string | null)
# Let's try POST to / with token=null explicitly
print("\n=== POST / with token:null ===")
r = post_json(API_BASE, {
    'api_name': 'fund_basic',
    'token': None,
    'params': {'market': 'E'},
    'fields': 'ts_code,name'
})
print(json.dumps(r, indent=2)[:500])

# Try POST to / with token in params
print("\n=== POST / with token in params ===")
for test_token in ['', 'guest', 'public', 'free']:
    r = post_json(API_BASE, {
        'api_name': 'fund_basic',
        'token': test_token,
        'params': {'market': 'E'},
        'fields': 'ts_code,name'
    })
    print(f"  token='{test_token}': {json.dumps(r, indent=2)[:200]}")

# Check /openapi.json for info on how the root POST works
print("\n=== Root POST schema from openapi.json ===")
oapi = json.loads(urllib.request.urlopen(API_BASE + 'openapi.json').read())
paths = oapi['paths']
root_post = paths['/'].get('post', {})
print(f"Summary: {root_post.get('summary', 'N/A')}")
print(f"RequestBody: {json.dumps(root_post.get('requestBody', {}), indent=2)[:1000]}")
print(f"Parameters: {json.dumps(root_post.get('parameters', []), indent=2)[:500]}")
