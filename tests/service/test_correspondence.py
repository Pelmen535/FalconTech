"""Correspondences are optional diagnostics, never an identity decision."""
import io
import subprocess
import numpy as np
import pytest
from PIL import Image
from service.correspondence import match_patches, draw_pairs, run_isolated
from service.robust import ResourceUnavailableError


def test_matching_recovers_permuted_parts_and_rejects_ambiguity():
    features = np.eye(576, dtype=np.float32)
    permutation = np.random.default_rng(4).permutation(576)
    pairs = match_patches(features, features[permutation])
    assert len(pairs) == 8
    for pair in pairs:
        assert permutation[pair['candidate_index']] == pair['query_index']
    assert match_patches(np.zeros((576,3)), np.zeros((576,3))) == []
    assert match_patches(np.ones((576,3)), np.ones((576,3))) == []


def test_match_rejects_nonfinite_or_incompatible_shapes():
    good = np.ones((576,3))
    for bad in [np.ones((575,3)), np.ones((576,4)), np.full((576,3),np.nan)]:
        with pytest.raises(ValueError):
            match_patches(good,bad)


def test_renderer_bounds_extreme_aspect_ratios_and_empty_matches():
    images = [Image.new('RGB',(1,9000)),Image.new('RGB',(9000,1))]
    with Image.open(io.BytesIO(draw_pairs(images,[]))) as result:
        result.load()
        assert result.width == 1200 and result.height <= 520


def test_resource_rejection_and_timeout_are_retryable_and_child_has_no_cuda(tmp_path,monkeypatch):
    import service.correspondence as module
    policy = {'minimum_headroom_GiB':3,'timeout_seconds':120}
    spec = {'operations':{'start_C_disk_free_GiB':5},'correspondence_policy':policy,'cpu_environment':{'checkpoint':'weight'},
            'files':{'weight':{'bytes':1,'SHA256':'a'*64}}}
    monkeypatch.setattr(module,'memory_headroom',lambda:2)
    with pytest.raises(ResourceUnavailableError):
        run_isolated(tmp_path,spec,'q',[0,0,1,1],'g',None)
    monkeypatch.setattr(module,'memory_headroom',lambda:8)
    def timeout(*args,**kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']==''
        assert kwargs['timeout']==120
        raise subprocess.TimeoutExpired(args[0],120)
    monkeypatch.setattr(module.subprocess,'run',timeout)
    with pytest.raises(ResourceUnavailableError):
        run_isolated(tmp_path,spec,'q',[0,0,1,1],'g',None)
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a,1,stderr=b'error'))
    with pytest.raises(ResourceUnavailableError):
        run_isolated(tmp_path,spec,'q',[0,0,1,1],'g',None)


def test_map_honors_existing_write_disk_reserve(tmp_path,monkeypatch):
    import service.disk_reserve as disk
    def reject(self):
        raise disk.DiskReserveError({'reasons':['temporary filesystem below reserve']})
    monkeypatch.setattr(disk.WriteDiskReserve,'require',reject)
    with pytest.raises(ResourceUnavailableError,match='below reserve'):
        run_isolated(tmp_path,{'correspondence_policy':{},
            'operations':{'start_C_disk_free_GiB':5}},'q',None,'g',None)
