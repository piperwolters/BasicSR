import torch
import clip
import open_clip
import torch.nn.functional as F

from basicsr.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_clipscore(img, img2, clip_model='clipa', **kwargs):
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    # Any CLIP model can be used to extract image features; assure the correct image size is used.
    # TODO: incorporate model-specific preprocessing (ex. normalization).
    if clip_model == 'clipa':
        img_size = (224,224)
        model, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14-CLIPA-336', pretrained='datacomp1b')
    else:
        print("Currently the only CLIP models supported are ['clipa'].")

    img = torch.transpose(torch.from_numpy(img), (2, 1, 0)).unsqueeze(0)
    img2 = torch.transpose(torch.from_numpy(img2), (2, 1, 0)).unsqueeze(0)

    img = F.interpolate(img, img_size)
    img2 = F.interpolate(img2, img_size)

    img1_feats = model.encode_image(img)
    img2_feats = model.encode_image(img2)

    sim_score = F.cosine_similarity(img1_feats, img2_feats)
    return sim_score
