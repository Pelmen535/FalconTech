"""Safety and representation regressions; quality is measured on saved captures."""
import pytest
import numpy as np
from PIL import Image
from service.refusal_guard import contradictory_rescue
from service.october_images import occlusion_regions
from service.explain import heatmap_png,top_regions
from io import BytesIO

def test_final_guard_preserves_earlier_accepts_and_requires_all_evidence():
    # Same weak pair with separately stronger evidence must remain eligible.
    result={'legacy_absent_probability':np.array([.85,.85,.79,.85,.85,.85]),
            'top':np.zeros(6,dtype=int),'raw':np.array([[.43],[.43],[.43],[.43],[.46],[.43]])}
    previous={'accepted':np.array([False,True,False,False,False,False])}
    current={'accepted':np.array([True,True,True,True,True,False])}
    score=np.array([.75,.75,.75,.81,.75,.75])
    actual=contradictory_rescue(result,previous,current,score)
    np.testing.assert_array_equal(actual,[True,False,False,False,False,False])
    permutation=np.array([5,2,0,4,3,1])
    reordered=contradictory_rescue({k:v[permutation] for k,v in result.items()},
        {k:v[permutation] for k,v in previous.items()},
        {k:v[permutation] for k,v in current.items()},score[permutation])
    np.testing.assert_array_equal(reordered,actual[permutation])
    for i in range(6):
        single=contradictory_rescue({k:v[i:i+1] for k,v in result.items()},
            {k:v[i:i+1] for k,v in previous.items()},
            {k:v[i:i+1] for k,v in current.items()},score[i:i+1])
        assert single[0]==actual[i]

@pytest.mark.parametrize('grid',[4,8])
def test_occlusion_masks_one_region_without_accumulating_or_changing_source(grid):
    pixels=np.full((13,19,3),230,dtype=np.uint8);source=Image.fromarray(pixels)
    masks=list(occlusion_regions(source,grid));changed=np.zeros((13,19),dtype=int)
    assert len(masks)==grid*grid
    for masked in masks:
        array=np.asarray(masked)
        assert set(np.unique(array))=={127,230}
        changed+=(array[:,:,0]!=pixels[:,:,0])
    # Every source pixel is changed exactly once across all independent masks.
    np.testing.assert_array_equal(changed,np.ones_like(changed))
    np.testing.assert_array_equal(np.asarray(source),pixels)

def test_signed_occlusion_map_does_not_center_away_uniform_positive_effect():
    source=Image.new('RGB',(32,32),(128,128,128))
    contributions=np.ones((4,4),dtype=np.float32)*.05
    with Image.open(BytesIO(heatmap_png(source,contributions,center=False))) as image:
        pixel=np.asarray(image)[16,16]
    assert pixel[1]>pixel[0] and pixel[1]>pixel[2]
    assert top_regions(contributions,1,center=False)[0]['contribution']>0
