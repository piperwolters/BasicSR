import clip
import open_clip
import torch.nn.functional as F

from basicsr.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_clipscore(img, img2, clip_model='clipa', **kwargs):
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    print("imgs:", img.shape, img2.shape, clip_model)
    if clip_model == 'clipa':
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14-CLIPA-336', pretrained='datacomp1b')

    img = F.interpolate(img, (224,224))
    img2 = F.interpolate(img2, (224,224))

    img1_feats = model(img)
    img2_feats = model(img2)

    sim_score = F.cosine_similarity(img1_feats, img2_feats)
    return sim_score
