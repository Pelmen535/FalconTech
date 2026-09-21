import csv
from pathlib import Path

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


def test_submission_is_written_in_the_official_headerless_format(tmp_path):
    """Формат submission.csv задан эталоном организаторов, а не нашим прочтением ТЗ.

    В шапке organizer/evaluate.py написано «Формат submission.csv (без заголовка)», и в
    organizer/example_submission.zip заголовка нет. Строка заголовка не ломает эталон, но
    разбирается им как лишний запрос с десятью неизвестными идентификаторами. Этот тест
    существует, чтобы заголовок не вернулся обратно незаметно.
    """
    import csv as _csv
    from vreid.artifacts import read_submission
    from vreid.submit import write_submission

    q_keys = [f"q{i}" for i in range(3)]
    g_keys = [f"g{i}" for i in range(12)]
    sims = np.arange(len(q_keys) * len(g_keys), dtype=np.float32).reshape(len(q_keys), -1)
    path = tmp_path / "submission.csv"
    write_submission(q_keys, g_keys, sims, path, topk=10, exclude_self=False)

    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(_csv.reader(stream))
    assert rows[0][0] == "q0", "первая строка обязана быть данными, а не заголовком"
    assert len(rows) == len(q_keys)
    assert all(len(row) == 11 for row in rows)

    assert [key for key, _ in read_submission(path)] == q_keys

    # Читатель обязан понимать и файлы прежних версий, где заголовок был.
    with path.open(encoding="utf-8", newline="") as stream:
        body = stream.read()
    legacy = tmp_path / "legacy.csv"
    legacy.write_text("query_id," + ",".join(f"gallery_id_{i}" for i in range(1, 11)) + "\n" + body,
                      encoding="utf-8")
    assert read_submission(legacy) == read_submission(path)


def test_official_example_submission_parses_with_our_reader():
    """Пример организаторов должен читаться нашим же читателем без оговорок."""
    import zipfile
    from vreid.artifacts import read_submission

    archive = Path(__file__).resolve().parents[1] / "organizer/example_submission.zip"
    if not archive.is_file():
        pytest.skip("organizer/example_submission.zip не приложен")
    with zipfile.ZipFile(archive) as zf:
        text = zf.read("submission.csv").decode("utf-8")
    tmp = Path(__file__).resolve().parents[1] / "results/_example_submission.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text, encoding="utf-8")
    try:
        # В примере галерея из восьми записей, поэтому и кандидатов восемь, а не десять:
        # «ровно десять» — это про наш набор, а не про формат файла.
        rows = read_submission(tmp, topk=8)
        assert len(rows) == 5
        assert rows[0][0] == "q_a1b2c3d4" and len(rows[0][1]) == 8
    finally:
        tmp.unlink(missing_ok=True)
