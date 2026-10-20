"""External API/schema/log contract; tiny encoder, no model quality claims."""
from dataclasses import replace
from io import BytesIO
import json
import re

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from service.app import create_app
from service.config import Settings
from service.schemas import (HealthResponse, GalleryItemResponse, GalleryResponse,
                             ModelResponse, IndexResponse, SearchResponse, ErrorResponse)


class Core:
    dimension = 4
    model_sha256 = 'a'*64
    recipe_sha256 = 'b'*64
    supports_ann = False
    confidence_scale = 'calibrated_acceptance_score'

    def __init__(self, accepted=True, failure=False):
        self.accepted, self.failure = accepted, failure

    def metadata(self):
        return {'loaded': True, 'threshold': .7, 'custom_core_marker': 'preserved'}

    def encode(self, path, bbox):
        if self.failure:
            raise RuntimeError('sensitive-internal-message')
        return np.array([1,0,0,0], np.float32)

    def rank(self, q, g, ids, query_id=None):
        ranking=[{'image_id':v,'rank':i+1,'cosine':.6-i*.01} for i,v in enumerate(ids[:10])]
        return dict(ranking=ranking, accepted=self.accepted, candidate=ranking[0] if self.accepted else None,
                    confidence=.8 if self.accepted else .5, threshold=.7, custom_core_marker='preserved')


def image():
    f=BytesIO();Image.new('RGB',(16,16),(10,20,30)).save(f,format='PNG');return f.getvalue()


def config(tmp_path, **kwargs):
    return replace(Settings(tmp_path,tmp_path,tmp_path/'data'), **kwargs)


def add(client, i, headers=None):
    return client.post('/v1/gallery/items',headers=headers,files={'image':('private-image.png',image(),'image/png')},
                       data={'bbox':'[0,0,16,16]','image_id':f'private-id-{i}'})


def events(caplog):
    return [json.loads(r.message) for r in caplog.records if r.name=='vreid.http']


def test_request_id_error_logging_and_no_payload_leak(tmp_path, caplog):
    caplog.set_level('INFO',logger='vreid.http')
    with TestClient(create_app(config(tmp_path,api_key='private-api-secret'),Core())) as c:
        ok=c.get('/health',headers={'X-Request-ID':'forged-client-id'})
        denied=c.get('/v1/model?private-query=secret-value')
        assert ok.status_code==200 and denied.status_code==401
        for response in [ok,denied]:
            rid=response.headers['X-Request-ID'];assert re.fullmatch('[0-9a-f]{32}',rid)
            row=next(e for e in events(caplog) if e['event']=='http.request' and e['request_id']==rid)
            assert row['status']==response.status_code and row['duration_ms']>=0
        logged=json.dumps(events(caplog));assert all(v not in logged for v in ['private-api-secret','secret-value','forged-client-id'])
        ErrorResponse.model_validate(denied.json())


def test_real_payloads_match_documented_schemas_without_filtering(tmp_path,caplog):
    caplog.set_level('INFO',logger='vreid.http');core=Core()
    with TestClient(create_app(config(tmp_path),core)) as c:
        HealthResponse.model_validate(c.get('/health').json());ModelResponse.model_validate(c.get('/v1/model').json());IndexResponse.model_validate(c.get('/v1/index/stats').json())
        for i in range(10):
            r=add(c,i);assert r.status_code==201;GalleryItemResponse.model_validate(r.json())
        GalleryResponse.model_validate(c.get('/v1/gallery').json())
        for accepted in [True,False]:
            core.accepted=accepted
            r=c.post('/v1/search',files={'image':('private-query.png',image(),'image/png')},data={'bbox':'[0,0,16,16]'})
            assert r.status_code==200;value=r.json();SearchResponse.model_validate(value)
            assert value['accepted'] is accepted and (value['candidate'] is None)==(not accepted)
            assert value['custom_core_marker']=='preserved' and len(value['ranking'])==10
            rid=r.headers['X-Request-ID'];rows=[e for e in events(caplog) if e['request_id']==rid]
            assert {e['event'] for e in rows}=={'http.request','model.operation','search.result'}
            assert any(e.get('phase')=='query.encode' for e in rows) and any(e.get('phase')=='search.rank' for e in rows)
            assert next(e for e in rows if e['event']=='search.result')['accepted'] is accepted
        c.get('/v1/gallery/items/private-id-0/image?private-query=secret-value')
        logged=json.dumps(events(caplog));assert all(v not in logged for v in ['private-id','private-image.png','private-query.png','secret-value'])


def test_unexpected_error_logged_without_sensitive_message(tmp_path,caplog):
    caplog.set_level('INFO',logger='vreid.http')
    with TestClient(create_app(config(tmp_path),Core(failure=True)),raise_server_exceptions=False) as c:
        r=add(c,0);assert r.status_code==500
        entries=events(caplog);assert any(e['event']=='http.request' and e['status']==500 and e['error_type']=='RuntimeError' for e in entries)
        assert any(e.get('phase')=='gallery.encode' and e['outcome']=='error' for e in entries)
        assert 'sensitive-internal-message' not in json.dumps(entries)
        assert c.get('/health').status_code==200


def test_openapi_success_errors_auth_and_bbox(tmp_path):
    with TestClient(create_app(config(tmp_path,api_key='never-publish-this'),Core())) as c:
        schema=c.get('/openapi.json').json();paths=schema['paths'];search=paths['/v1/search']['post']
        assert schema['components']['securitySchemes']['ApiKeyAuth']=={'type':'apiKey','in':'header','name':'X-API-Key'}
        assert search['security']==[{'ApiKeyAuth':[]}]
        assert 'security' not in paths['/health']['get']
        for path,definitions in paths.items():
            for method,definition in definitions.items():
                if method not in ['get','post','delete']:continue
                success='204' if method=='delete' else '201' if method=='post' and path in ['/v1/gallery/items','/v1/gallery/import','/v1/demo/gallery'] else '200'
                assert 'X-Request-ID' in definition['responses'][success]['headers']
        assert search['responses']['200']['content']['application/json']['schema']['$ref'].endswith('/SearchResponse')
        assert 'Retry-After' in search['responses']['503']['headers']
        body=search['requestBody']['content']['multipart/form-data']['schema']['$ref'].split('/')[-1]
        assert 'width,height' in schema['components']['schemas'][body]['properties']['bbox']['description']
        assert 'never-publish-this' not in json.dumps(schema)
        assert 'content' not in paths['/v1/gallery/items/{image_id}']['delete']['responses']['204']


def test_openapi_auth_not_required_when_unconfigured(tmp_path):
    with TestClient(create_app(config(tmp_path),Core())) as c:
        assert 'security' not in c.get('/openapi.json').json()['paths']['/v1/model']['get']
