import copy
import uuid
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select
from collector.config import Settings
from collector.api import create_app
from collector.db import Administrator, Base, Device, Token


def login_admin(system: dict) -> TestClient:
    with system['app'].state.sessions.begin() as session:
        session.add(Administrator(user_id=system['owner']))
    # Secure cookies are sent only through HTTPS, including in tests.
    client = TestClient(system['app'], base_url='https://testserver')
    response = client.post('/admin/login', json={'username': 'alice', 'password': 'correct-password-123'})
    assert response.status_code == 200
    assert 'HttpOnly' in response.headers['set-cookie']
    assert 'Secure' in response.headers['set-cookie']
    assert 'SameSite=strict' in response.headers['set-cookie']
    client.headers['X-CSRF-Token'] = response.json()['csrf']
    return client


def test_admin_auth_is_required_and_cookie_is_not_api_auth(system: dict) -> None:
    assert system['client'].get('/admin/users', headers=system['headers']).status_code == 401
    client = login_admin(system)
    assert client.get('/admin/users').status_code == 200
    assert client.get('/v1/runs').status_code == 401
    assert client.post('/admin/logout').status_code == 200
    assert client.get('/admin/users').status_code == 401


def login_participant(system: dict) -> TestClient:
    client = TestClient(system['app'], base_url='https://testserver')
    response = client.post('/admin/login', json={'username':'bob', 'password':'correct-password-456'})
    assert response.status_code == 200
    assert response.json()['administrator'] is False
    client.headers['X-CSRF-Token'] = response.json()['csrf']
    return client


def test_participant_web_access_is_scoped(system: dict, payload: dict) -> None:
    admin = login_admin(system)
    client = login_participant(system)
    assert client.get('/admin/session').json()['administrator'] is False
    assert [u['id'] for u in client.get('/admin/users').json()] == [system['other']]
    assert client.get('/admin/users/'+system['owner']+'/devices').status_code == 403
    assert client.get('/admin/users/'+system['other']+'/devices').json() == []
    assert client.patch('/admin/users/'+system['owner'], json={'active': False}).status_code == 403
    assert client.post('/admin/users/'+system['owner']+'/revoke').status_code == 403
    assert client.patch('/admin/users/'+system['other']+'/devices/'+str(uuid.uuid4()), json={'label':'x', 'active':False}).status_code == 403
    assert client.post('/account/users', json={'username':'child', 'password':'new-password-123'}, headers={'X-CSRF-Token':''}).status_code == 403
    assert client.post('/account/users', json={'username':'child', 'password':'new-password-123','creator_id':system['owner']}).status_code == 422
    made = client.post('/account/users', json={'username':'Child', 'password':'new-password-123'})
    assert made.status_code == 201
    child = made.json()['id']
    assert client.patch('/admin/users/'+child, json={'password':'other-password-123'}).status_code == 403
    assert client.get('/admin/users/'+child+'/devices').status_code == 403
    history = client.get('/account/creations').json()
    assert len(history) == 1 and history[0]['username'] == 'child' and history[0]['created_ms'] > 0
    assert admin.get('/account/creations').json() == []
    recorded = next(u for u in admin.get('/admin/users').json() if u['id'] == child)
    assert recorded['created_by'] == 'bob' and recorded['created_ms'] == history[0]['created_ms']
    assert recorded['administrator'] is False
    nested = TestClient(system['app'], base_url='https://testserver')
    auth = nested.post('/admin/login', json={'username':'child','password':'new-password-123'})
    nested.headers['X-CSRF-Token'] = auth.json()['csrf']
    assert nested.post('/account/users',json={'username':'grandchild','password':'new-password-456'}).status_code == 201
    assert [u['username'] for u in client.get('/account/creations').json()] == ['child']
    assert client.post('/account/users', json={'username':'child','password':'new-password-123'}).status_code == 409
    assert len(client.get('/account/creations').json()) == 1
    assert client.post('/admin/logout').status_code == 200
    assert client.get('/account/creations').status_code == 401


def test_creation_quota_and_disabled_creator(system: dict) -> None:
    from collector.db import AccountCreation, User
    from collector.security import now_ms
    admin = login_admin(system)
    client = login_participant(system)
    with system['app'].state.sessions.begin() as session:
        for i in range(20):
            uid = str(uuid.uuid4())
            session.add(User(id=uid,username=f'quota{i}',password_hash='unused',active=True))
            session.flush()
            session.add(AccountCreation(user_id=uid,creator_id=system['other'],created_ms=now_ms()))
    response = client.post('/account/users', json={'username':'overquota','password':'new-password-123'})
    assert response.status_code == 429 and 'Retry-After' in response.headers
    with system['app'].state.sessions.begin() as session:
        for row in session.scalars(select(AccountCreation)):
            row.created_ms = now_ms() - 86400001
    assert client.post('/account/users', json={'username':'allowed','password':'new-password-123'}).status_code == 201
    assert admin.patch('/admin/users/'+system['other'],json={'active':False}).status_code == 200
    assert client.post('/account/users',json={'username':'blocked','password':'new-password-123'}).status_code == 401


def test_csrf_and_cross_site_login(system: dict) -> None:
    client = login_admin(system)
    response = client.post('/admin/users', json={'username':'new', 'password':'new-password-123'}, headers={'X-CSRF-Token':''})
    assert response.status_code == 403
    assert client.post('/admin/login', json={'username':'alice','password':'correct-password-123'}, headers={'Sec-Fetch-Site':'cross-site'}).status_code == 403


def test_create_reset_disable_user_and_revoke(system: dict) -> None:
    client = login_admin(system)
    body = {'username':'Charlie', 'password':'new-password-123'}
    result=client.post('/admin/users',json=body)
    assert result.status_code == 201
    uid=result.json()['id']
    assert client.post('/admin/users',json=body).status_code == 409
    auth=system['client'].post('/v1/auth/login',json={**body,'username':'charlie'}).json()
    headers={'Authorization':'Bearer '+auth['token']}
    assert client.patch('/admin/users/'+uid,json={'password':'changed-password-456'}).status_code == 200
    assert system['client'].get('/v1/runs',headers=headers).status_code == 401
    assert client.patch('/admin/users/'+uid,json={'active':False}).status_code == 200
    assert system['client'].post('/v1/auth/login',json={'username':'charlie','password':'changed-password-456'}).status_code == 401
    assert client.patch('/admin/users/'+system['owner'],json={'active':False}).status_code == 409


def test_multi_device_stats_and_disable(system: dict, payload: dict) -> None:
    client=login_admin(system)
    assert system['client'].post('/v1/ingest',headers=system['headers'],json=payload).status_code==200
    second=copy.deepcopy(payload)
    second['run']['id']=str(uuid.uuid4())
    second['run']['device_id']=str(uuid.uuid4())
    second['run']['device_model']='Second phone'
    assert system['client'].post('/v1/ingest',headers=system['headers'],json=second).status_code==200
    path='/admin/users/'+system['owner']+'/devices'
    devices=client.get(path).json()
    assert len(devices)==2
    assert all(d['samples']==1 for d in devices)
    assert {d['model'] for d in devices}=={'test-device','Second phone'}
    user=next(u for u in client.get('/admin/users').json() if u['id']==system['owner'])
    assert user['devices']==2 and user['samples']==2
    assert client.patch(path+'/'+payload['run']['device_id'],json={'label':'Work phone','active':False}).status_code==200
    assert system['client'].post('/v1/ingest',headers=system['headers'],json=payload).status_code==403
    assert system['client'].post('/v1/ingest',headers=system['headers'],json=second).status_code==200
    assert client.get(path).json()[0]['label']=='Work phone'


def test_same_device_id_is_scoped_by_user(system: dict, payload: dict) -> None:
    client=login_admin(system)
    assert system['client'].post('/v1/ingest',headers=system['headers'],json=payload).status_code==200
    second=copy.deepcopy(payload)
    second['run']['id']=str(uuid.uuid4())
    assert system['client'].post('/v1/ingest',headers=system['other_headers'],json=second).status_code==200
    with system['app'].state.sessions() as session:
        assert len(list(session.scalars(select(Device))))==2
    for uid in [system['owner'],system['other']]:
        assert client.get('/admin/users/'+uid+'/devices').json()[0]['samples']==1


def test_prefix_and_static_assets(tmp_path: Path) -> None:
    app=create_app(Settings(database_url=f'sqlite:///{tmp_path}/prefix.db',root_path='/projects/uiembeddings'))
    Base.metadata.create_all(app.state.engine)
    with TestClient(app,base_url='https://testserver') as client:
        page=client.get('/projects/uiembeddings/')
        assert page.status_code==200
        assert 'assets/admin.js' in page.text
        assert client.get('/projects/uiembeddings/assets/admin.js').status_code==200
        assert client.get('/projects/uiembeddings/assets/unknown').status_code==404
        assert "frame-ancestors 'none'" in page.headers['content-security-policy']
        assert client.get('/projects/uiembeddings/healthz').json()=={'status':'ok'}


def test_calibration_only_device_is_visible_and_disable_blocks_report(system: dict) -> None:
    client=login_admin(system)
    report=dict(id=str(uuid.uuid4()),device_id=str(uuid.uuid4()),model_id=system['model_id'],wall_ms=1790000000000,backend='litert_opencl_full_encoder',gpu_parity_cosine=1.,selected_fps=2.5,thermal_status=0,results=[dict(target_fps=2.5,duration_ms=30000.,processed=75,fresh_frames=75,p50_ms=10.,p95_ms=12.,achieved_fps=2.5,missed_deadlines=0,sustainable=True)])
    assert system['client'].post('/v1/benchmarks',headers=system['headers'],json=report).status_code==200
    path='/admin/users/'+system['owner']+'/devices'
    d=client.get(path).json()[0]
    assert d['runs']==0 and d['samples']==0 and d['calibration']['selected_fps']==2.5
    assert client.patch(path+'/'+d['id'],json={'label':'Calibration phone','active':False}).status_code==200
    assert system['client'].post('/v1/benchmarks',headers=system['headers'],json=report).status_code==403


def test_creation_quota_is_atomic(system: dict) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from collector.db import AccountCreation, User
    from collector.security import now_ms
    client = login_participant(system)
    with system['app'].state.sessions.begin() as session:
        for i in range(19):
            uid = str(uuid.uuid4())
            session.add(User(id=uid,username=f'previous{i}',password_hash='unused',active=True))
            session.flush()
            session.add(AccountCreation(user_id=uid,creator_id=system['other'],created_ms=now_ms()))
    def create(i: int) -> int:
        return client.post('/account/users', json={'username':f'concurrent{i}','password':'new-password-123'}).status_code
    with ThreadPoolExecutor(max_workers=4) as pool:
        codes = list(pool.map(create, range(4)))
    assert sorted(codes) == [201, 429, 429, 429]
    assert len(client.get('/account/creations').json()) == 20


def test_additive_creation_migration_preserves_users(system: dict) -> None:
    from collector.db import AccountCreation, User
    engine = system['app'].state.engine
    AccountCreation.__table__.drop(engine)
    AccountCreation.__table__.create(engine, checkfirst=True)
    AccountCreation.__table__.create(engine, checkfirst=True)
    with system['app'].state.sessions() as session:
        assert set(session.scalars(select(User.id))) == {system['owner'], system['other']}
        assert list(session.scalars(select(AccountCreation))) == []
