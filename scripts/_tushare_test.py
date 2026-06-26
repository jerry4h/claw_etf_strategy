#!/usr/bin/env python3
"""Quick Tushare API connectivity test."""
import urllib.request
import json
import sys

API_BASE = 'http://8.148.76.181:8686/'

def test_connectivity():
    """Test basic connectivity to the Tushare API."""
    try:
        req = urllib.request.Request(API_BASE)
        resp = urllib.request.urlopen(req, timeout=10)
        body = resp.read().decode('utf-8', errors='replace')
        print(f"Status: {resp.status}")
        print(f"Headers: {dict(resp.headers)}")
        print(f"Body (first 1000 chars): {body[:1000]}")
        
        # Try to parse as JSON
        try:
            data = json.loads(body)
            print(f"\nParsed JSON keys: {list(data.keys()) if isinstance(data, dict) else 'list'}")
        except json.JSONDecodeError:
            print(f"\nBody is not JSON. Content type: {resp.headers.get('Content-Type', 'unknown')}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

if __name__ == '__main__':
    test_connectivity()
