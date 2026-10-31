"""Explicit downloads, separated from the offline demo command."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request
from PIL import Image

DATASET = 'timm/oxford-iiit-pet'
BREEDS = ('bengal', 'british_shorthair', 'siamese', 'sphynx', 'beagle', 'pug', 'shiba_inu', 'samoyed')
MODEL_URL = 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt'
MODEL_SHA256 = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'

def read_url(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return response.read()

def select_rows(split, count):
    chosen = {breed: [] for breed in BREEDS}
    for offset in range(0, 1000, 100):
        query = urllib.parse.urlencode(dict(dataset=DATASET, config='default', split=split, offset=offset, length=100))
        result = json.loads(read_url('https://datasets-server.huggingface.co/rows?' + query))
        names = next(feature['type']['names'] for feature in result['features'] if feature['name']=='label')
        for item in result['rows']:
            row = item['row']
            breed = names[row['label']]
            if breed in chosen and len(chosen[breed]) < count:
                chosen[breed].append((row, item['row_idx'], split))
        if all(len(items)==count for items in chosen.values()):
            return [item for breed in BREEDS for item in chosen[breed]]
    raise RuntimeError('Insufficient photos for frozen selection')

def prepare(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    images = root / 'images'
    images.mkdir(exist_ok=True)
    manifest_path = root / 'catalog.json'
    if not manifest_path.exists():
        selected = [('catalog', item) for item in select_rows('train', 4)] + [('query', item) for item in select_rows('test', 1)]
        def save(item):
            kind, (row, row_index, split) = item
            image_id = row['image_id']
            if not re.fullmatch(r'[A-Za-z0-9_]+', image_id):
                raise ValueError('Unsafe image ID')
            original = read_url(row['image']['src'])
            photo = Image.open(io.BytesIO(original)).convert('RGB')
            photo.thumbnail((384,384))
            target = images / (image_id+'.jpg')
            photo.save(target, quality=88)
            return dict(id=image_id, image='images/'+target.name, label=image_id.rsplit('_',1)[0].lower(), kind=kind, split=split,
                        source=f'https://huggingface.co/datasets/{DATASET}/viewer/default/{split}?row={row_index}',
                        sha256=hashlib.sha256(target.read_bytes()).hexdigest(), original_sha256=hashlib.sha256(original).hexdigest(),
                        license='CC-BY-SA-4.0', changes='Resized to fit384x384 and re-encoded JPEG')
        with ThreadPoolExecutor(max_workers=6) as pool:
            records = list(pool.map(save, selected))
        ids = [item['id'] for item in records]
        if len(ids)!=len(set(ids)) or len({item['original_sha256'] for item in records})!=len(records):
            raise ValueError('Duplicate catalog/query images')
        manifest_path.write_text(json.dumps(records, indent=2)+'\n', encoding='utf-8')
    weights = root.parent / '.cache' / 'ViT-B-32.pt'
    weights.parent.mkdir(parents=True, exist_ok=True)
    if not weights.exists():
        temporary = weights.with_suffix('.partial')
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, temporary.open('wb') as output:
            while chunk := response.read(1024*1024):
                output.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest()!=MODEL_SHA256:
            raise ValueError('Unexpected CLIP checkpoint hash')
        temporary.replace(weights)
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=MODEL_SHA256:
        raise ValueError('CLIP checkpoint hash mismatch')
    return {'catalog':str(manifest_path), 'weights':str(weights)}
