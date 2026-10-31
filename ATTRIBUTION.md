# Photo and model attribution

## Oxford-IIIT Pet photographs

The small photo subset comes from the Oxford-IIIT Pet dataset by O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar, described in *Cats and Dogs*, CVPR 2012.

- Official dataset and license statement: https://www.robots.ox.ac.uk/~vgg/data/pets/
- License: Creative Commons Attribution-ShareAlike 4.0 International, https://creativecommons.org/licenses/by-sa/4.0/
- Mirror used for individual downloads: https://huggingface.co/datasets/timm/oxford-iiit-pet
- Copyright remains with the original image owners, as stated on the official dataset page.

Each retained photo has its dataset image ID, split, source-row permalink, source/content hash, and license in `artifacts/catalog.json`. The copies are resized to fit 384 by 384 pixels and re-encoded as JPEG. The copied/adapted photos remain under CC BY-SA 4.0. These image terms do not purport to license the entire portfolio codebase. Preserve this attribution and the per-photo manifest when redistributing the image subset.

## Pretrained encoder

The demo runs the genuine OpenAI CLIP ViT-B/32 checkpoint locally through OpenCLIP, with no fabricated embedding fallback.

- Official model card: https://huggingface.co/openai/clip-vit-base-patch32
- Original implementation/license: https://github.com/openai/CLIP/blob/main/LICENSE (MIT, copyright 2021 OpenAI)
- OpenCLIP implementation/license: https://github.com/mlfoundations/open_clip (MIT)
- Checkpoint source: https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
- Verified checkpoint SHA256: `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`

Model weights are downloaded into an ignored local cache and are not bundled into the portfolio. The model card discusses research use and the need for task-specific testing. This demonstration uses a fixed eight-breed taxonomy, reports its actual retrieval failures and baselines, and does not claim general or production suitability.
