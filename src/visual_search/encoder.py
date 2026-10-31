from pathlib import Path
import numpy as np
from PIL import Image
from .index import normalize

class ClipEncoder:
    def __init__(self,weights:Path):
        if not weights.exists():
            raise FileNotFoundError('Run python run.py prepare before inference')
        import torch
        import open_clip
        torch.set_num_threads(2)
        self.torch = torch
        from open_clip.transform import image_transform
        self.model = open_clip.load_openai_model(str(weights),device='cpu',precision='fp32')
        self.preprocess = image_transform(224,is_train=False,mean=(0.48145466,0.4578275,0.40821073),std=(0.26862954,0.26130258,0.27577711))
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

    def images(self,paths):
        result=[]
        with self.torch.inference_mode():
            for start in range(0,len(paths),8):
                tensors=[]
                for path in paths[start:start+8]:
                    with Image.open(path) as image:
                        tensors.append(self.preprocess(image.convert('RGB')))
                result.append(self.model.encode_image(self.torch.stack(tensors)).float().cpu().numpy())
        return normalize(np.concatenate(result))

    def texts(self,texts):
        with self.torch.inference_mode():
            return normalize(self.model.encode_text(self.tokenizer(texts)).float().cpu().numpy())

def histogram(path):
    with Image.open(path) as image:
        rgb=np.asarray(image.convert('RGB').resize((64,64)))
    vector=np.concatenate([np.histogram(rgb[:,:,channel],bins=16,range=(0,256))[0] for channel in range(3)]).astype(np.float32)
    return normalize(vector.reshape(1,-1))[0]

def lexical(query,records):
    words=set(query.lower().replace('_',' ').split())
    scores=[]
    for item in records:
        label_words=set(item['label'].replace('_',' ').split())
        scores.append({'id':item['id'],'score':len(words & label_words)/max(1,len(label_words))})
    return sorted(scores,key=lambda item:(-item['score'],item['id']))
