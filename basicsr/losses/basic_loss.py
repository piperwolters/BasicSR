import torch
import clip
import open_clip
import torchvision
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss

@LOSS_REGISTRY.register()
class CLIPLoss(nn.Module):
    
    def __init__(self, clip_loss_model):
        super(CLIPLoss, self).__init__()
        
        if clip_loss_model == 'EVA02-E-14-plus':
            self.sim_model, _, _ = open_clip.create_model_and_transforms('EVA02-E-14-plus', pretrained='laion2b_s9b_b144k')
        elif clip_loss_model == 'ViT-B-16-SigLIP-256'
            self.sim_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16-SigLIP-256', pretrained='webli')
        elif clip_loss_model == 'RN50':
            self.sim_model, _ = clip.load("RN50")

        self.normalize = Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)

    def forward(self, x, gt):
        x = F.interpolate(x, (224, 224))
        gt = F.interpolate(gt, (224, 224))

        x = self.normalize(x)
        gt = self.normalize(gt)
        print("range:", torch.min(x), torch.max(x), torch.min(gt), torch.max(gt))

        x_feats = self.sim_model.encode_image(x)
        gt_feats = self.sim_model.encode_image(gt)
        l1 = l1_loss(x_feats, gt_feats)
        return l1

@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1',
                 model_weights=None):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights

        # Specific case when we pass in pretrained ResNet weights to use instead of VGG.
        if vgg_type == 'caco_resnet':
            print("Utilizing a pretrained ResNet50 for perceptual loss.")
            perceptual_weights = torch.load(model_weights) #torch.load('weights/resnet50_caco_1m.pth')
            self.resnet = torchvision.models.resnet50()
            self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-2])).cuda()

            new_keys = {}
            perceptual_keys = list(perceptual_weights.keys())
            for i,resnet_key in enumerate(self.resnet.state_dict().keys()):
                if i == 0:
                    continue
                pk = perceptual_keys[i]
                new_keys[resnet_key] = None
                new_keys[resnet_key] = perceptual_weights[pk]

            self.resnet.load_state_dict(new_keys, strict=False)
            self.layer_weights = [0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1]
        elif vgg_type == 'rrsgan_vgg19':
            print("Utilizing a remote-sensing pretrained VGG19 for perceptual loss.")
            perceptual_weights = torch.load(model_weights)  # weights/vgg19-dcbb9e9d.pth

            self.vgg = VGGFeatureExtractor(
                layer_name_list=list(layer_weights.keys()),
                vgg_type='vgg19',
                use_input_norm=use_input_norm,
                range_norm=range_norm)

            new_keys = {}
            perceptual_keys = list(perceptual_weights.keys())
            for i,resnet_key in enumerate(self.vgg.vgg_net.state_dict().keys()):
                pk = perceptual_keys[i]
                new_keys[resnet_key] = None
                new_keys[resnet_key] = perceptual_weights[pk]

            self.vgg.vgg_net.load_state_dict(new_keys)
        else:
            self.vgg = VGGFeatureExtractor(
                layer_name_list=list(layer_weights.keys()),
                vgg_type=vgg_type,
                use_input_norm=use_input_norm,
                range_norm=range_norm)
        self.vgg_type = vgg_type

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        if self.vgg_type == 'caco_resnet':
            x_features, gt_features = {}, {}

            # Save feature maps at each layer for both the ground truth and generated output.
            self.layer_name_list = list(self.resnet.state_dict().keys())
            for key, layer in self.resnet._modules.items():
                x = layer(x)
                gt = layer(gt)
                x_features[key] = x.clone()
                gt_features[key] = gt.clone()

            if self.perceptual_weight > 0:
                # calculate perceptual loss
                percep_loss = 0
                for i,k in enumerate(['0', '1', '2', '3', '4', '5', '6', '7']):
                    if self.criterion_type == 'fro':
                        percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[i]
                    else:
                        percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[i]
                percep_loss *= self.perceptual_weight
            else:
                percep_loss = None

            # calculate style loss
            if self.style_weight > 0:
                style_loss = 0
                for i,k in enumerate(['0', '1', '2', '3', '4', '5', '6', '7']):
                    if self.criterion_type == 'fro':
                        style_loss += torch.norm(
                            self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                    else:
                        style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                            gt_features[k])) * self.layer_weights[k]
                style_loss *= self.style_weight
            else:
                style_loss = None
        else:
            # extract vgg features
            x_features = self.vgg(x)
            gt_features = self.vgg(gt.detach())

            # calculate perceptual loss
            if self.perceptual_weight > 0:
                percep_loss = 0
                for k in x_features.keys():
                    if self.criterion_type == 'fro':
                        percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                    else:
                        percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
                percep_loss *= self.perceptual_weight
            else:
                percep_loss = None

            # calculate style loss
            if self.style_weight > 0:
                style_loss = 0
                for k in x_features.keys():
                    if self.criterion_type == 'fro':
                        style_loss += torch.norm(
                            self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                    else:
                        style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                            gt_features[k])) * self.layer_weights[k]
                style_loss *= self.style_weight
            else:
                style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
