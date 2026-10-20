"""Backup includes live WAL content; restore validates before publishing."""
from pathlib import Path
import json,sqlite3,zipfile
import numpy as np
import pytest
from service.gallery_bundle import create_bundle,restore_bundle,ReadOnlySQLite
from service.storage import GalleryStore,GalleryIdentityError


def store_at(path):return GalleryStore(path,'a'*64,'b'*64,2)


def test_snapshot_cold_restore_preserves_order_images_boxes_vectors(tmp_path):
    store=store_at(tmp_path/'source')
    store.add('first',b'original-1',(1,2,10,20),np.array([.6,.8],np.float32))
    store.add('second',b'original-2',(2,3,20,30),np.array([0.,1.],np.float32))
    bundle=tmp_path/'backup.zip';receipt=create_bundle(ReadOnlySQLite(tmp_path/'source'),bundle)
    assert receipt['observations']==2
    target=tmp_path/'restored';restore_bundle(bundle,target)
    restored=store_at(target)
    a,av=store.snapshot();b,bv=restored.snapshot()
    assert a==b and np.array_equal(av,bv) and store.list_items()==restored.list_items()
    assert restored.image('second')==b'original-2'
    with pytest.raises(FileExistsError):restore_bundle(bundle,target)
    with pytest.raises(FileExistsError):create_bundle(store,bundle)


def test_tampered_image_leaves_no_active_restore(tmp_path):
    store=store_at(tmp_path/'source');store.add('one',b'original',(0,0,1,1),np.array([1.,0.]))
    bundle=tmp_path/'backup.zip';create_bundle(store,bundle)
    bad=tmp_path/'bad.zip'
    with zipfile.ZipFile(bundle) as src,zipfile.ZipFile(bad,'w') as dst:
        for name in src.namelist():dst.writestr(name,b'changed!' if name.startswith('images/') else src.read(name))
    with pytest.raises(ValueError,match='mismatch'):restore_bundle(bad,tmp_path/'not-published')
    assert not (tmp_path/'not-published').exists() and store.count()==1
    assert not list(tmp_path.glob('gallery-restore-*'))


def test_wrong_model_and_unsafe_member_fail_before_activation(tmp_path):
    store=store_at(tmp_path/'source');bundle=tmp_path/'backup.zip';create_bundle(store,bundle)
    with pytest.raises(GalleryIdentityError):restore_bundle(bundle,tmp_path/'wrong',expected_identity={'model_sha256':'c'*64})
    with zipfile.ZipFile(bundle,'a') as archive:archive.writestr('raw/C:/outside',b'no')
    with pytest.raises(ValueError):restore_bundle(bundle,tmp_path/'unsafe')
    assert not (tmp_path/'wrong').exists() and not (tmp_path/'unsafe').exists()
