import json
from pathlib import Path
import numpy as np
import pytest
from visual_search.index import CosineIndex,normalize
from visual_search.metrics import check_split,evaluate_ranking
from visual_search.encoder import histogram,lexical
from PIL import Image

def records():
    return [{'id':'a','label':'red cat'},{'id':'b','label':'blue dog'},{'id':'c','label':'green cat'}]

def test_cosine_normalizes_queries_and_vectors_and_breaks_ties_by_id():
    index=CosineIndex(records(),[[3,0],[0,9],[3,0]])
    assert [item['id'] for item in index.search([10,0],3)]==['a','c','b']
    assert index.search([10,0],1)[0]['score']==pytest.approx(1)

@pytest.mark.parametrize('vectors',[[[0,0]],[[float('nan'),1]],[[float('inf'),1]]])
def test_invalid_embeddings_fail_closed(vectors):
    with pytest.raises(ValueError):
        normalize(vectors)

def test_index_rejects_duplicate_ids_misaligned_dimensions_and_bad_k():
    with pytest.raises(ValueError):
        CosineIndex([{'id':'a'},{'id':'a'}],[[1,0],[0,1]])
    index=CosineIndex(records(),[[1,0],[0,1],[1,1]])
    with pytest.raises(ValueError): index.search([1,0,0])
    with pytest.raises(ValueError): index.search([1,0],0)

def test_updates_are_idempotent_and_removal_restores_original_content():
    index=CosineIndex(records(),[[1,0],[0,1],[1,1]])
    before=index.digest
    new={'id':'d','label':'purple pet'}
    index.upsert(new,[0.4,0.6])
    after=index.digest
    assert after!=before and len(index.records)==4
    assert index.search([.4,.6])[0]['id']=='d'
    index.upsert(new,[0.4,0.6])
    assert index.digest==after and len(index.records)==4
    index.remove('d')
    assert index.digest==before and len(index.records)==3

def test_serialization_preserves_content_hash_and_rankings(tmp_path):
    index=CosineIndex(records(),[[.2,.3],[.5,.8],[1,.3]])
    target=tmp_path/'index.npz'
    index.save(target)
    loaded=CosineIndex.load(target)
    assert loaded.digest==index.digest
    assert loaded.search([.1,.2])==index.search([.1,.2])

def test_metrics_account_for_missed_relevant_items_and_bad_rankings():
    scores=evaluate_ranking(['x','a','y','b','z'],['a','b','c'])
    assert scores['precision_at_5']==pytest.approx(.4)
    assert scores['recall_at_5']==pytest.approx(2/3)
    assert scores['mrr']==pytest.approx(.5)
    assert 0<scores['ndcg_at_5']<1
    with pytest.raises(ValueError): evaluate_ranking(['a','a'],['a'])
    with pytest.raises(ValueError): evaluate_ranking(['a'],[])

def test_held_out_ids_and_pixels_cannot_enter_catalog():
    with pytest.raises(ValueError): check_split([{'id':'x','sha256':'1'}],[{'id':'x','sha256':'2'}])
    with pytest.raises(ValueError): check_split([{'id':'x','sha256':'1'}],[{'id':'y','sha256':'1'}])
    check_split([{'id':'x','sha256':'1'}],[{'id':'y','sha256':'2'}])

def test_histogram_baseline_reads_actual_pixels(tmp_path):
    red,blue=tmp_path/'red.png',tmp_path/'blue.png'
    Image.new('RGB',(40,40),(255,0,0)).save(red)
    Image.new('RGB',(40,40),(0,0,255)).save(blue)
    assert not np.allclose(histogram(red),histogram(blue))
    assert np.linalg.norm(histogram(red))==pytest.approx(1)

def test_metadata_baseline_matches_labels_without_model_calls():
    assert lexical('a red cat',records())[0]['id']=='a'

def test_recorded_real_evidence_has_disjoint_photos_and_complete_ranking_contract():
    root=Path(__file__).resolve().parents[1]/'artifacts'
    if not (root/'demo.json').exists(): pytest.skip('Run prepare and demo for real inference evidence')
    report=json.loads((root/'demo.json').read_text(encoding='utf-8'))
    catalog_ids={item['id'] for item in report['catalog']}
    assert len(catalog_ids)==32
    assert report['index']['dimensions']==512
    for query in report['queries']:
        assert query['query_image_id'] not in catalog_ids
        for method in ('clip','baseline'):
            assert len(query['results'][method])==8
            assert all(item['id'] in catalog_ids for item in query['results'][method])
    assert report['update']['restored_original_hash']
