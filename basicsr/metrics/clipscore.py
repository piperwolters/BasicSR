import clip
import open_clip
import torch.nn.functional as F

@METRIC_REGISTRY.register()
def calculate_clipscore(img1, img2, clip_model):
    print("imgs:", img1.shape, img2.shape, clip_model)
    if clip_model == 'clipa':
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14-CLIPA-336', pretrained='datacomp1b')

    

    img1_feats = model(img1)
    img2_feats = model(img2)

    sim_score = F.cosine_similarity(img1_feats, img2_feats)
    return sim_score
