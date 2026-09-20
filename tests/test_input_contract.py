import csv
import numpy as np
import pytest
from PIL import Image
from vreid.hackathon_data import read_annotations
from vreid.extract import crop_bbox
from vreid.submit import write_submission,save_embeddings

def test_strict_parser_rejects_duplicate_frame(tmp_path):
    p=tmp_path/'input.csv'
    with p.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['image_id','x','y','w','h']);w.writerows([['x',0,0,10,10]]*2)
    with pytest.raises(ValueError,match='unique'):read_annotations(p,tmp_path,strict=True)
    assert len(read_annotations(p,tmp_path))==2  # legacy multi-object research input is explicit

@pytest.mark.parametrize('box', [(0, 0, 0, 2), (0, 0, float('nan'), 2)])
def test_invalid_bbox_raises(box):
    """Нечисло или нулевая сторона — это мусор во входе, осмысленного кропа нет."""
    with pytest.raises(ValueError):
        crop_bbox(Image.new('RGB', (20, 20)), box, pad=0)


def test_bbox_outside_frame_is_clamped_not_substituted(capsys):
    """ББox корректен, но вне кадра.

    Три варианта поведения, и все три плохие по-разному:
      весь кадр — тихая подмена объекта, никто не заметит;
      падение — ноль за всю сдачу из-за одной строки CSV;
      прижатие к границе — плохой эмбеддинг ровно у одного запроса.
    Выбран третий: строка embedding обязательна для каждого image_id (ответ 49),
    а молчания нет — в лог уходит предупреждение."""
    im = Image.new('RGB', (20, 20))
    crop = crop_bbox(im, (100, 100, 2, 2), pad=0)
    assert crop.size != im.size, 'кроп не должен подменяться всем кадром'
    assert crop.size[0] >= 1 and crop.size[1] >= 1
    assert 'вне кадра' in capsys.readouterr().out

def test_small_gallery_is_not_padded(tmp_path):
    out=tmp_path/'submission.csv'
    with pytest.raises(ValueError):write_submission(['q'],['a','b'],np.ones((1,2)),out)
    assert not out.exists()

def test_nonfinite_embeddings_cannot_be_saved(tmp_path):
    with pytest.raises(ValueError):save_embeddings(np.array([[np.nan,1]],np.float32),tmp_path/'e.npy')

def test_negative_infinity_is_blocked_candidate(tmp_path):
    with pytest.raises(ValueError):
        write_submission(['q'],list(map(str,range(10))),np.array([[-np.inf]+[0.]*9]),tmp_path/'s.csv')
