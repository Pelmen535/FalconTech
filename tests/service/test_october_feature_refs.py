"""Observation refs survive persistence, slicing and legacy additive migration."""
import sqlite3
import numpy as np
import pytest
from service.storage import GalleryStore,GalleryVectors
from service.october_features import FeatureVector,RawFeatureStore,snapshot_signature


def test_feature_refs_are_observation_specific_and_survive_restart(tmp_path):
    store=GalleryStore(tmp_path,'model','embedding',512)
    vector=np.zeros(512,np.float32);vector[0]=1
    store.add('one',b'first-frame',(0,0,10,10),FeatureVector(vector,'a'*64))
    store.add('two',b'second-frame',(0,0,10,10),FeatureVector(vector,'b'*64))
    ids,values=GalleryStore(tmp_path,'model','embedding',512).snapshot()
    assert ids==['one','two'] and values.feature_refs==('a'*64,'b'*64)
    assert values[[1,0]].feature_refs==('b'*64,'a'*64)
    assert values[:1].feature_refs==('a'*64,)
    assert store.image('one')==b'first-frame' and store.image('two')==b'second-frame'
    large=np.tile(vector,(10,1));keys=[str(i) for i in range(10)];refs=['a'*64]*9+['b'*64]
    assert snapshot_signature(large,keys,refs)!=snapshot_signature(large,keys,refs[::-1])
    store.delete('one');assert store.snapshot()[1].feature_refs==('b'*64,)


def test_additive_old_schema_keeps_images_vectors_and_identity(tmp_path):
    store=GalleryStore(tmp_path,'m','r',2);store.add('old',b'original',(0,0,1,1),np.array([1.,0.]))
    # Reproduce the original schema without the additive column.
    with sqlite3.connect(store.db_path) as conn:
        conn.execute('ALTER TABLE gallery_items DROP COLUMN feature_ref')
    restored=GalleryStore(tmp_path,'m','r',2);ids,v=restored.snapshot()
    assert ids==['old'] and v.feature_refs==(None,) and restored.image('old')==b'original'
    assert np.array_equal(v,np.array([[1.,0.]],np.float32))


def test_distinct_raw_refs_with_same_projection_are_explicit_and_legacy_fails(tmp_path):
    store=RawFeatureStore(tmp_path,'identity');raw=np.zeros(2048,np.float32);cnn=np.ones(2048,np.float16)
    base=np.zeros(512,np.float32);base[0]=1
    first=store.put(raw,cnn,base,persist=True,explicit_ref=True)
    second=store.put(raw,cnn*2,base,persist=True,explicit_ref=True)
    assert first!=second and np.array_equal(store.get(second,persisted=True)['specialist'],cnn*2)
    with pytest.raises(ValueError,match='Ambiguous'):store.legacy(base,persisted=True)


def test_invalid_reference_batch_rolls_back(tmp_path):
    store=GalleryStore(tmp_path,'m','r',2)
    with pytest.raises(ValueError,match='feature_ref'):
        store.add_many([dict(image_id='bad',image_bytes=b'frame',bbox_xywh=(0,0,1,1),embedding=np.array([1.,0.]),feature_ref='../outside')])
    assert store.count()==0
