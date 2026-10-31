import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from .encoder import ClipEncoder, histogram, lexical
from .index import CosineIndex
from .metrics import check_split, evaluate_ranking
from .prepare import MODEL_SHA256

SPECIES = {'bengal':'cat','british_shorthair':'cat','siamese':'cat','sphynx':'cat','beagle':'dog','pug':'dog','shiba_inu':'dog','samoyed':'dog'}

def content(root):
    artifacts=root/'artifacts'
    records=json.loads((artifacts/'catalog.json').read_text(encoding='utf-8'))
    for item in records:
        if hashlib.sha256((artifacts/item['image']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Catalog image changed; prepare a new frozen manifest')
    catalog=[item for item in records if item['kind']=='catalog']
    held_out=[item for item in records if item['kind']=='query']
    check_split(catalog,held_out)
    with (root/'.cache/ViT-B-32.pt').open('rb') as file:
        if hashlib.file_digest(file,'sha256').hexdigest()!=MODEL_SHA256:
            raise ValueError('Checkpoint differs from verified source')
    return artifacts,catalog,held_out

def summary(records,method):
    values=[item['scores'][method] for item in records]
    keys=('precision_at_5','recall_at_5','ndcg_at_5','mrr')
    return {**{key:float(np.mean([item[key] for item in values])) for key in keys},
            'mean_latency_ms':float(np.mean([item['latency_ms'][method] for item in records])),
            'queries':len(records)}

def demo(root:Path):
    artifacts,catalog,held_out=content(root)
    started=time.perf_counter()
    encoder=ClipEncoder(root/'.cache/ViT-B-32.pt')
    initialization_ms=(time.perf_counter()-started)*1000
    paths=[artifacts/item['image'] for item in catalog]
    started=time.perf_counter()
    vectors=encoder.images(paths)
    index=CosineIndex(catalog,vectors)
    build_ms=(time.perf_counter()-started)*1000
    index_path=root/'.cache/index.npz'
    index.save(index_path)
    color_index=CosineIndex(catalog,np.stack([histogram(path) for path in paths]))
    # Warm both encoders before diagnostic per-query timing.
    encoder.texts(['a pet photograph'])
    encoder.images([artifacts/held_out[0]['image']])
    results=[]
    for item in held_out:
        for kind in ('text','image'):
            text=f"a photo of a {item['label'].replace('_',' ')} {SPECIES[item['label']]}"
            started=time.perf_counter()
            vector=encoder.texts([text])[0] if kind=='text' else encoder.images([artifacts/item['image']])[0]
            ranked=index.search(vector,k=len(catalog))
            clip_ms=(time.perf_counter()-started)*1000
            started=time.perf_counter()
            baseline=lexical(text,catalog) if kind=='text' else color_index.search(histogram(artifacts/item['image']),k=len(catalog))
            baseline_ms=(time.perf_counter()-started)*1000
            relevant=[row['id'] for row in catalog if row['label']==item['label']]
            results.append({'id':f"{kind}-{item['id']}",'kind':kind,'text':text,
                            'query_image':item['image'] if kind=='image' else None,'query_image_id':item['id'],
                            'relevant_ids':relevant,'results':{'clip':ranked[:8],'baseline':baseline[:8]},
                            'scores':{'clip':evaluate_ranking([row['id'] for row in ranked],relevant),
                                      'baseline':evaluate_ranking([row['id'] for row in baseline],relevant)},
                            'latency_ms':{'clip':clip_ms,'baseline':baseline_ms}})
    original_hash=index.digest
    inserted=dict(held_out[0])
    new_vector=encoder.images([artifacts/inserted['image']])[0]
    before=len(index.records)
    index.upsert(inserted,new_vector)
    after=len(index.records)
    changed=index.digest!=original_hash
    top=index.search(new_vector,k=1)[0]['id']
    index.upsert(inserted,new_vector)
    idempotent=len(index.records)
    index.remove(inserted['id'])
    update={'before_count':before,'after_count':after,'idempotent_count':idempotent,'restored_count':len(index.records),
            'new_item_top_result':top,'version_changed':changed,'restored_original_hash':index.digest==original_hash,
            'note':'Update executed only after evaluation; held-out query image is inserted then removed and never contributes to reported metrics.'}
    artifact={'version':'0.1','catalog':catalog,'queries':results,
              'metrics':{method:summary(results,method) for method in ('clip','baseline')},
              'by_kind':{kind:{method:summary([item for item in results if item['kind']==kind],method) for method in ('clip','baseline')} for kind in ('text','image')},
              'index':{'version':'0.1','dimensions':int(vectors.shape[1]),'count':len(catalog),'bytes':index_path.stat().st_size,'sha256':original_hash},
              'update':update,
              'provenance':{'model':'OpenAI CLIP ViT-B/32','checkpoint_sha256':MODEL_SHA256,'model_card':'https://huggingface.co/openai/clip-vit-base-patch32',
                            'dataset':'Oxford-IIIT Pet subset via timm mirror','dataset_url':'https://www.robots.ox.ac.uk/~vgg/data/pets/',
                            'dataset_license':'CC-BY-SA-4.0','catalog_split':'32 train images, four per breed','query_split':'8 separate test images, one per breed',
                            'relevance':'Dataset breed labels are a constrained proxy, not independently collected human search judgments.',
                            'baseline':'Metadata token overlap for text; 48-dimensional RGB histogram cosine for image queries.',
                            'index_build_ms':build_ms,'model_initialization_ms':initialization_ms,'python':platform.python_version(),'device':'CPU; two PyTorch threads',
                            'image_changes':'Resized to fit384x384; JPEG re-encoding. Rights remain with original owners; see ATTRIBUTION.md.'},
              'limitations':['A fixed eight-breed, 32-photo catalog is a constrained research demonstration, not a general visual search service.',
                             'Breed agreement is only a relevance proxy; no independent human search-quality study was performed.',
                             'Catalog and query IDs/bytes are disjoint, but original CLIP pretraining overlap with Oxford Pets is unknown.',
                             'Text queries name the breed; metadata lexical matching has privileged labels and is a strong baseline. Image baseline uses only color histograms.',
                             'CLIP scores are cosine similarities, not calibrated probabilities. Top matches are returned even for irrelevant queries.',
                             'Local warmed single-query timings include query encoding and exact search, but exclude model initialization, HTTP, concurrent load, and image downloading.',
                             'OpenAI CLIP model card describes research scope; no unrestricted production suitability is claimed.']}
    (artifacts/'demo.json').write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return {'artifact':str(artifacts/'demo.json'),'metrics':artifact['metrics'],'by_kind':artifact['by_kind'],'update':update}

def query(root:Path,text):
    if not text or not text.strip():
        raise ValueError('Supply a nonempty --text query')
    content(root)
    index=CosineIndex.load(root/'.cache/index.npz')
    encoder=ClipEncoder(root/'.cache/ViT-B-32.pt')
    started=time.perf_counter()
    result=index.search(encoder.texts([text])[0])
    return {'text':text,'results':result,'latency_ms':(time.perf_counter()-started)*1000,'index_sha256':index.digest,
            'scope':'Fixed pet catalog; results are similarities, including for out-of-domain requests.'}
