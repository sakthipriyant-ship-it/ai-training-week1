import os
import hmac
import hashlib

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()
API_KEY = os.getenv("API_KEY")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

session = requests.Session()
retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retry))
session.mount("http://", HTTPAdapter(max_retries=retry))

response = session.get("https://jsonplaceholder.typicode.com/users", timeout=10)
response.raise_for_status()
users = response.json()

for user in users[:5]:
    print(f"{user.get('id')} - {user.get('name')} - {user.get('email')}")

first = users[0]
address = first.get("address", {})
geo = address.get("geo", {})
company = first.get("company", {})

print(f"City: {address.get('city', 'Not Available')} | Zip: {address.get('zip', 'Not Available')}")

def verify_signature(payload, signature, secret):
    expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


payload = '{"event": "user.created", "id": 1}'
valid_sig = hmac.new(WEBHOOK_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()

valid = verify_signature(payload, valid_sig, WEBHOOK_SECRET)
tampered = verify_signature('{"event": "user.created", "id": 2}', valid_sig, WEBHOOK_SECRET)
wrong_secret = verify_signature(payload, valid_sig, "wrong_secret")

print(f"Valid: {valid} | Tampered: {tampered} | Wrong secret: {wrong_secret}")