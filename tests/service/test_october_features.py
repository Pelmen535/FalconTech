"""Raw cache integrity/privacy; these fixtures do not claim model quality."""
import numpy as np
import pytest
from service.october_features import RawFeatureStore


def values():
 rng=np.random.default_rng(81)
 raw=rng.normal(size=2048).astype(np.float32);raw/=np.linalg.norm(raw)
 cnn=rng.normal(size=2048).astype(np.float16)
 base=rng.normal(size=512).astype(np.float32);base/=np.linalg.norm(base)
 return raw,cnn,base


def test_queries_are_memory_only_and_gallery_legacy_rollback_survives_restart(tmp_path):
 store=RawFeatureStore(tmp_path,'identity');raw,cnn,base=values()
 ref=store.put(raw,cnn,base)
 assert not list(tmp_path.rglob('*.npz'))
 fresh=RawFeatureStore(tmp_path,'identity')
 with pytest.raises(ValueError,match='not available'):fresh.get(ref)
 assert store.put(raw,cnn,base,persist=True)==ref
 restarted=RawFeatureStore(tmp_path,'identity');got,packet=restarted.legacy(base,persisted=True)
 assert got==ref and np.array_equal(packet['emb'],raw)
 with np.load(tmp_path/(store.token(base)+'.npz')) as legacy:
  assert np.array_equal(legacy['base'],base) and np.array_equal(legacy['specialist'],cnn)


def test_corrupt_objects_and_wrong_model_identity_fail_closed(tmp_path):
 store=RawFeatureStore(tmp_path,'identity');raw,cnn,base=values();ref=store.put(raw,cnn,base,persist=True)
 with pytest.raises(ValueError,match='integrity'):RawFeatureStore(tmp_path,'other-model').get(ref,persisted=True)
 raw=raw.copy();raw[0]+=.1
 np.savez(store.objects/(ref+'.npz'),emb=raw,specialist=cnn,base=base)
 with pytest.raises(ValueError,match='integrity'):store.get(ref,persisted=True)


def test_same_projection_different_raw_does_not_silently_overwrite_legacy_lookup(tmp_path):
 store=RawFeatureStore(tmp_path,'identity');raw,cnn,base=values();ref=store.put(raw,cnn,base,persist=True)
 other=cnn.copy();other[0]+=1
 with pytest.raises(ValueError,match='Ambiguous'):store.put(raw,other,base,persist=True)
 assert store.legacy(base,persisted=True)[0]==ref


def test_nan_dimensions_and_path_injection_are_rejected(tmp_path):
 store=RawFeatureStore(tmp_path,'identity');raw,cnn,base=values()
 with pytest.raises(ValueError):store.put(raw[:1],cnn,base)
 invalid=raw.copy();invalid[0]=np.nan
 with pytest.raises(ValueError):store.put(invalid,cnn,base)
 with pytest.raises(ValueError):store.get('../outside')


def test_gallery_ingestion_cache_is_content_bbox_and_model_specific(tmp_path):
 image=tmp_path/'same-name.jpg';image.write_bytes(b'fixture-source-bytes')
 store=RawFeatureStore(tmp_path/'cache','identity');raw,cnn,base=values()
 key=store.input_key(image,[0,0,10,10]);assert store.find_input(key) is None
 ref=store.put(raw,cnn,base,persist=True);assert store.remember_input(key,ref)[0]==ref
 cold=RawFeatureStore(tmp_path/'cache','identity');assert cold.find_input(key)[0]==ref
 assert cold.find_input(cold.input_key(image,[1,0,10,10])) is None
 assert RawFeatureStore(tmp_path/'cache','new-model').input_key(image,[0,0,10,10])!=key
 image.write_bytes(b'new-source-content');assert cold.input_key(image,[0,0,10,10])!=key
 with pytest.raises(ValueError):cold.find_input('../outside')


def test_persisted_wrong_dtype_is_not_silently_reinterpreted(tmp_path):
 store=RawFeatureStore(tmp_path,'identity');raw,cnn,base=values();ref=store.put(raw,cnn,base,persist=True)
 np.savez(store.objects/(ref+'.npz'),emb=raw.astype(np.float64),specialist=cnn,base=base)
 with pytest.raises(ValueError,match='typed'):store.get(ref,persisted=True)
