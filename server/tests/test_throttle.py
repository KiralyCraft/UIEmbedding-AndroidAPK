from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from collector.api import create_app
from collector.config import Settings
from collector.throttle import enforce_login_limit


def limited_app(system: dict, **overrides: int) -> FastAPI:
    return create_app(replace(Settings(database_url=system['app'].state.engine.url.render_as_string(hide_password=False),login_ip_burst_limit=3),**overrides))


def request_from(host: str) -> Request:
    return Request({'type':'http','client':(host,1234),'headers':[]})


def test_web_and_collector_share_ip_budget_and_sessions_keep_working(system: dict, payload: dict) -> None:
    app=limited_app(system)
    with TestClient(app,base_url='https://testserver') as client:
        for n,path in enumerate(('admin/login','v1/auth/login','admin/login')):
            assert client.post('/'+path,json={'username':f'missing-{n}','password':'invalid'}).status_code==401
        response=client.post('/v1/auth/login',json={'username':'alice','password':'correct-password-123'})
        assert response.status_code==429 and 1<=int(response.headers['retry-after'])<=60
        assert client.post('/v1/ingest',headers=system['headers'],json=payload).status_code==200
    # A fresh application instance uses the same database counters.
    with TestClient(limited_app(system),base_url='https://testserver') as fresh:
        assert fresh.post('/admin/login',json={'username':'other','password':'invalid'}).status_code==429
    with TestClient(limited_app(system),client=('198.51.100.22',1234)) as another_ip:
        assert another_ip.post('/v1/auth/login',json={'username':'missing-other','password':'invalid'}).status_code==401


def test_successful_login_does_not_reset_ip_budget(system: dict) -> None:
    with TestClient(limited_app(system)) as client:
        for _ in range(3):
            assert client.post('/v1/auth/login',json={'username':'alice','password':'correct-password-123'}).status_code==200
        assert client.post('/v1/auth/login',json={'username':'alice','password':'correct-password-123'}).status_code==429


def test_forged_forwarding_headers_do_not_change_client_budget(system: dict) -> None:
    app=ProxyHeadersMiddleware(limited_app(system),trusted_hosts=['192.168.2.67'])
    with TestClient(app,client=('192.168.2.67',1234)) as proxy:
        for n in range(4):
            # Apache appends the actual connection address after user-supplied values.
            response=proxy.post('/v1/auth/login',json={'username':f'unknown-{n}','password':'invalid'},headers={'X-Forwarded-For':f'203.0.113.{n+1}, 198.51.100.10'})
            assert response.status_code==(401 if n<3 else 429)
    with TestClient(app,client=('198.51.100.10',1234)) as direct:
        assert direct.post('/v1/auth/login',json={'username':'unknown-direct','password':'invalid'},headers={'X-Forwarded-For':'203.0.113.100'}).status_code==429


def test_atomic_admission_under_concurrent_attempts(system: dict) -> None:
    config=Settings(login_ip_burst_limit=3)
    def attempt(_: int) -> int:
        try:
            enforce_login_limit(request_from('198.51.100.40'),config,system['app'].state.sessions)
            return 200
        except HTTPException as exc:
            return exc.status_code
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes=list(pool.map(attempt,range(12)))
    assert outcomes.count(200)==3 and outcomes.count(429)==9, outcomes


def test_expiry_and_long_window(system: dict, monkeypatch) -> None:
    config=Settings(login_ip_burst_limit=2,login_ip_long_limit=3)
    now=1790000000000
    monkeypatch.setattr('collector.throttle.now_ms',lambda:now)
    request=request_from('198.51.100.50')
    for _ in range(2):
        enforce_login_limit(request,config,system['app'].state.sessions)
    now+=61000
    enforce_login_limit(request,config,system['app'].state.sessions)
    try:
        enforce_login_limit(request,config,system['app'].state.sessions)
        raise AssertionError('Long-window limit was bypassed')
    except HTTPException as exc:
        assert exc.status_code==429 and int(exc.headers['Retry-After'])==839
    now+=900000
    enforce_login_limit(request,config,system['app'].state.sessions)
