from flask import request
from functools import wraps

_cf_header = "CF-Connecting-IP"
_other_headers = ["cf-connecting-ip", "X-Real-IP", "X-Forwarded-For"]
def client_ip(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = ""
        cloudflare = request.headers.get(_cf_header,"")
        if not ip and cloudflare:
            ip = cloudflare
        for h in _other_headers:
            candidate = request.headers.get(h,"")
            if candidate:
                ip = candidate
                break
        if not ip:
            ip = request.remote_addr
        return f(client_ip=ip, *args, **kwargs)

    return decorated
