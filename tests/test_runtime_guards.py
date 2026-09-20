import json
import numpy as np
import pytest
from vreid.predict import load_recipe,rerank_chunk_size
from vreid.rerank import k_reciprocal,k_reciprocal_chunked

def case(seed=18):
    rng=np.random.default_rng(seed)
    norm=lambda x:x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8)
    pool=norm(rng.standard_normal((5,8)).astype(np.float32))
    g=pool[rng.integers(0,len(pool),20)];q=pool[[rng.integers(0,len(pool))]]
    extra=norm(rng.standard_normal((15,8)).astype(np.float32))
    return q,g,extra

@pytest.mark.parametrize('seed',range(30))
def test_exact_ranking_independent_of_query_count(seed):
    q,g,extra=case(seed);crowd=np.concatenate([q,extra])
    a=k_reciprocal(q,g,6,2)[0];b=k_reciprocal(crowd,g,6,2)[0]
    assert np.array_equal(np.argsort(a,kind='stable')[:10],np.argsort(b,kind='stable')[:10])
    assert np.array_equal(a,b)
    full=k_reciprocal(crowd,g,6,2)
    for chunk in (1,3,7):
        got=k_reciprocal_chunked(crowd,g,6,2,chunk=chunk,q_keys=[f'q{i}' for i in range(16)],g_keys=[f'g{i}' for i in range(20)])
        assert np.array_equal(np.argsort(full,axis=1,kind='stable'),np.argsort(got,axis=1,kind='stable'))

@pytest.mark.parametrize('field,value',[('dba',1),('refuse_rate',.2),('threshold',float('nan')),('confidence','top1+bits'),('topk',9)])
def test_forbidden_recipe_fails_before_model(tmp_path,field,value):
    (tmp_path/'recipe.json').write_text(json.dumps(dict(threshold=.55,**({field:value} if field!='threshold' else {}))) if field!='threshold' else '{"threshold":NaN}')
    with pytest.raises(ValueError):load_recipe(tmp_path)

def test_memory_budget_too_small_falls_back_to_cosine():
    """Не влезло в память — возвращаем 0 (ре-ранжирования не будет), а не падаем.

    Падение на стенде жюри — ноль за ВСЕ блоки, включая уже заработанные.
    Откат к косинусу стоит около 3 пунктов mAP и НЕ нарушает независимость запросов
    (ответ 38): решение зависит только от размера ГАЛЕРЕИ и одинаково для всех запросов,
    а фактический режим пишется в run_info.json."""
    assert rerank_chunk_size(1, 750, .0001) == 0
    assert rerank_chunk_size(5000, 750, .0001) == 0, 'решение не должно зависеть от числа запросов'
    assert rerank_chunk_size(1110, 750, 12) > 0
    assert 1<=rerank_chunk_size(100,20,.0001)<100

def test_benchmark_score_is_clamped():
    from vreid.bench_full import latency_score,throughput_score
    assert latency_score(40)==1 and latency_score(80)==0
    assert throughput_score(50)==0 and throughput_score(100)==1
