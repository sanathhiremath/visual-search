"""Versioned exact-cosine index with deterministic ordering and explicit updates."""
import hashlib
import json
from pathlib import Path
import numpy as np

def normalize(vectors):
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError('Expected finite 2D embeddings')
    lengths = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(lengths <= 1e-10):
        raise ValueError('Zero embeddings are not searchable')
    return array / lengths

class CosineIndex:
    def __init__(self, records, vectors):
        self.records = list(records)
        self.vectors = normalize(vectors)
        if not self.records or len(self.records) != len(self.vectors):
            raise ValueError('Nonempty aligned catalog/embedding rows required')
        self._check_ids()

    def _check_ids(self):
        ids = [item['id'] for item in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate catalog IDs')

    @property
    def digest(self):
        rows = sorted(range(len(self.records)), key=lambda row:self.records[row]['id'])
        h = hashlib.sha256(json.dumps([self.records[row] for row in rows],sort_keys=True).encode())
        h.update(self.vectors[rows].astype('<f4').tobytes())
        return h.hexdigest()

    def search(self, vector, k=8):
        if k < 1:
            raise ValueError('k must be positive')
        query = normalize(np.asarray(vector).reshape(1,-1))[0]
        if query.shape[0] != self.vectors.shape[1]:
            raise ValueError('Embedding dimensions do not match')
        scores = self.vectors @ query
        positions = sorted(range(len(scores)), key=lambda row:(-float(scores[row]),self.records[row]['id']))[:k]
        return [{'id':self.records[row]['id'],'score':float(scores[row])} for row in positions]

    def upsert(self, record, vector):
        row = normalize(np.asarray(vector).reshape(1,-1))[0]
        if row.shape[0] != self.vectors.shape[1]:
            raise ValueError('Embedding dimensions do not match')
        position = next((index for index,item in enumerate(self.records) if item['id']==record['id']),None)
        if position is None:
            self.records.append(dict(record))
            self.vectors = np.vstack([self.vectors,row])
        else:
            self.records[position] = dict(record)
            self.vectors[position] = row

    def remove(self, identity):
        position = next((index for index,item in enumerate(self.records) if item['id']==identity),None)
        if position is None:
            raise KeyError(identity)
        if len(self.records)==1:
            raise ValueError('Cannot empty the index')
        del self.records[position]
        self.vectors = np.delete(self.vectors,position,axis=0)

    def save(self, path: Path):
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(path,vectors=self.vectors,records=np.asarray(json.dumps(self.records)))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            records=json.loads(str(data['records']))
            vectors=data['vectors'].copy()
        normalize(vectors)  # Validate finite, nonzero rows without changing saved bytes.
        if not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5):
            raise ValueError('Saved index vectors are not normalized')
        if not records or len(records)!=len(vectors):
            raise ValueError('Invalid saved index rows')
        obj=cls.__new__(cls)
        obj.records,obj.vectors=records,vectors
        obj._check_ids()
        return obj
