import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import time
import csv
import os

from model.ops import resize
from torch.special import expm1
from einops import rearrange, repeat
from mmcv.cnn import ConvModule

from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
from torchvision.transforms import ToPILImage

 
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))

def alpha_cosine_log_snr(t, ns=0.0002, ds=0.00025):
    # not sure if this accounts for beta being clipped to 0.999 in discrete version
    return -log((torch.cos((t + ns) / (1 + ds) * math.pi * 0.5) ** -2) - 1, eps=1e-5)

def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))
           
class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

class DepthwiseConvBlock(nn.Module):
    """深度可分卷积 + BN + ReLU"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class OCTEncoder(nn.Module):
    def __init__(self, in_channels=128, out_channels=256, num_blocks=3):
        super().__init__()
        layers = []
        # 第一层直接投影通道
        layers.append(DepthwiseConvBlock(in_channels, out_channels))
        # 中间残差卷积块
        for _ in range(num_blocks - 1):
            layers.append(nn.Sequential(
                DepthwiseConvBlock(out_channels, out_channels),
                DepthwiseConvBlock(out_channels, out_channels)
            ))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: [b, in_channels, H, W]
        return: [b, out_channels, H, W] 空间大小保持不变
        """
        return self.encoder(x)
    
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

class ClassHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=3, n_blocks=2, hidden_channels=128, dropout=0.3):
        super().__init__()

        layers = []
        # 输入层
        layers.append(DepthwiseConvBlock(in_channels, hidden_channels))
        # 堆叠 block
        for _ in range(n_blocks):
            layers.append(DepthwiseConvBlock(hidden_channels, hidden_channels))
        self.blocks = nn.Sequential(*layers)

        # 注意力增强
        self.att = SEBlock(hidden_channels)

        # 分类头 (MLP + Dropout + BN)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.att(x)
        x = self.pool(x).flatten(1)
        return self.mlp(x)

class CMFA(nn.Module):
    def __init__(self, channels=256, reduction=16):
        super(CMFA, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.fusion_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def channel_attention(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

    def spatial_attention(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.spatial_conv(attn))
        return x * attn

    def forward(self, fundus_feat, oct_feat):
        f_att = self.channel_attention(fundus_feat)
        o_att = self.channel_attention(oct_feat)
        fused = f_att + o_att
        fused = self.spatial_attention(fused)
        fused = F.relu(self.fusion_conv(fused))
        return fused
    
def focal_loss(logits, targets, gamma=2.0, weight=None):
    ce_loss = F.cross_entropy(logits, targets, weight=weight, reduction='none')
    pt = torch.exp(-ce_loss)
    loss = ((1 - pt) ** gamma * ce_loss).mean()
    return loss

def append_prediction_to_csv(filename, pred_class, csv_path="./out/Classification_Results.csv"):
    header = ["data", "non", "early", "mid_advanced"]

    # 如果文件不存在，先写入表头
    write_header = not os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(header)

        # 去掉扩展名作为 data
        data_name = os.path.splitext(filename)[0]

        # one-hot 编码
        row = [0, 0, 0]
        row[pred_class] = 1

        writer.writerow([data_name] + row)
    
@SEGMENTORS.register_module()
class IOMSG(EncoderDecoder):
    
    def __init__(self,
                 bit_scale=0.1,
                 timesteps=1,
                 randsteps=1,
                 time_difference=1,
                 learned_sinusoidal_dim=16,
                 sample_range=(0, 0.999),
                 noise_schedule='cosine',
                 diffusion='ddim',
                 accumulation=False,
                 **kwargs):
        super(IOMSG, self).__init__(**kwargs)

        self.bit_scale = bit_scale
        self.timesteps = timesteps
        self.randsteps = randsteps
        self.diffusion = diffusion
        self.time_difference = time_difference
        self.sample_range = sample_range
        self.accumulation = accumulation
        self.embedding_table = nn.Embedding(self.num_classes + 1, self.decode_head.in_channels[0])
        self.log_snr = alpha_cosine_log_snr
        self.transform = ConvModule(
            self.decode_head.in_channels[0] * 2,
            self.decode_head.in_channels[0],
            1,
            padding=0,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None
        )
        self.CMFA = ConvModule(
            self.decode_head.in_channels[0] * 2,
            self.decode_head.in_channels[0],
            1,
            padding=0,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None
        )
        # time embeddings
        time_dim = self.decode_head.in_channels[0] * 4  # 1024
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(  # [2,]
            sinu_pos_emb,  # [2, 17]
            nn.Linear(fourier_dim, time_dim),  # [2, 1024]
            nn.GELU(),
            nn.Linear(time_dim, time_dim)  # [2, 1024]
        )
        self.oct_encoder=OCTEncoder()
        self.class_head=ClassHead()
        

    def right_pad_dims_to(self, x, t):
        padding_dims = x.ndim - t.ndim
        if padding_dims <= 0:
            return t
        return t.view(*t.shape, *((1,) * padding_dims))

    def _get_sampling_timesteps(self, batch, *, device):
        times = []
        for step in range(self.timesteps):
            t_now = 1 - (step / self.timesteps) * (1 - self.sample_range[0])
            t_next = max(1 - (step + 1 + self.time_difference) / self.timesteps * (1 - self.sample_range[0]),
                         self.sample_range[0])
            time = torch.tensor([t_now, t_next], device=device)
            time = repeat(time, 't -> t b', b=batch)
            times.append(time)
        return times
    
    @torch.no_grad()
    def ddim_sample(self, feature,img_metas):
        b, c, h, w, device = *feature.shape, feature.device #[b,256,h/4,w/4]
        time_pairs = self._get_sampling_timesteps(b, device=device)
        feature = repeat(feature, 'b c h w -> (r b) c h w', r=self.randsteps)
        mask_t = torch.randn((self.randsteps, self.decode_head.in_channels[0], h, w), device=device)
        for idx, (times_now, times_next) in enumerate(time_pairs):
            feat = torch.cat([feature,mask_t], dim=1) #[b,512,h/4,w/4]
            feat = self.transform(feat)#[b,256,h/4,w/4]
            log_snr = self.log_snr(times_now)#[1]
            log_snr_next = self.log_snr(times_next)#[1]

            padded_log_snr = self.right_pad_dims_to(mask_t, log_snr) #pad log_snr [1]-[1,1,1,1]
            padded_log_snr_next = self.right_pad_dims_to(mask_t, log_snr_next) #pad log_snr [1]-[1,1,1,1]
            sigma, alpha = log_snr_to_alpha_sigma(padded_log_snr)
            sigma_next, alpha_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            input_times = self.time_mlp(log_snr)#1->[1,1024]
            mask_logit= self.decode_head.forward_test([feat], input_times)  # [bs, 256,h/4,w/4 ]-[b,9,h/4,w/4]
            mask_pred = torch.argmax(mask_logit, dim=1)#[b,1,h/4,w/4]
            """turn seg results to pred noise"""
            mask_pred = self.embedding_table(mask_pred).permute(0, 3, 1, 2)
            mask_pred = (torch.sigmoid(mask_pred) * 2 - 1) * self.bit_scale #scale to -0.01-0.01
            """epsilon_t=(x_t-sigma_t*x_t)/alpha_t"""
            pred_noise = (mask_t - sigma * mask_pred) / alpha.clamp(min=1e-8)
            """x_t-1=alpha_t-1*epsilon_t+sigma_t-1*x_t"""
            mask_t = alpha_next*pred_noise+sigma_next*mask_pred 
            
        logit = mask_logit.mean(dim=0, keepdim=True)
        return logit
    
    
    def encode_decode(self, img,ir, img_metas):
        """"""
        oct=ir.squeeze(1)
        feature_fundus = self.extract_feat(img)[0]#[b,256, h/4, w/4]
        """oct"""
        H,W=feature_fundus.shape[2:]
        oct=F.interpolate(oct, size=(H,W), mode='bilinear', align_corners=False)
        feat_oct=self.oct_encoder(oct)# in [b,128,256,256] out# bs, 256, h/4, w/4
        #feature = self.transform(torch.cat([feature_fundus, feat_oct], dim=1)) # (bs, 512, h/4, w/4) -> bs, 256, h/4, w/4
        feature=self.CMFA(torch.cat([feature_fundus, feat_oct], dim=1))
        """classification"""
        class_logits=self.class_head(feature) #[bs,3]
        pred_label =torch.argmax(class_logits, dim=1) # [bs,1]
       # append_prediction_to_csv(img_metas[0]['ori_filename'], pred_label)
        """"""
        out = self.ddim_sample(feature,img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        )
        return out,pred_label,img_metas[0]['ori_filename']

    def forward_train(self, img, img_metas, ir,gt_semantic_seg,label):
        oct=ir.squeeze(1)  # [b,128,256,256]
        label=label.squeeze()
        losses = dict()
        """image"""
        feature_fundus = self.extract_feat(img)[0]  # bs, 256, h/4, w/4
        """oct"""
        H,W=feature_fundus.shape[2:]
        oct=F.interpolate(oct, size=(H,W), mode='bilinear', align_corners=False)
        feat_oct=self.oct_encoder(oct)# in [b,128,H W] out# bs, 256,H W
        feature=self.CMFA(torch.cat([feature_fundus, feat_oct], dim=1)) # (bs, 256,H W,bs, 256,H W) ->bs, 256,H W
        #feature = self.transform(torch.cat([feature_fundus, feat_oct], dim=1)) # (bs, 512, h/4, w/4) -> bs, 256, h/4, w/4
        """classification"""
        class_logits=self.class_head(feature) #[bs,3]
        class_weights = torch.tensor([1.0, 1.0, 5.0]).to(label.device)
        loss_cls = focal_loss(class_logits, label, gamma=2.0, weight=class_weights)
        losses['loss_cls'] = loss_cls
        """gtdown represents the embedding of semantic segmentation labels after downsampling"""
        batch, c, h, w, device, = *feature.shape, feature.device
        gt_down = resize(gt_semantic_seg.float(), size=(h, w), mode="nearest")
        gt_down = gt_down.to(gt_semantic_seg.dtype)
        gt_down[gt_down == 255] = self.num_classes
        gt_down = self.embedding_table(gt_down).squeeze(1).permute(0, 3, 1, 2)
        gt_down = (torch.sigmoid(gt_down) * 2 - 1) * self.bit_scale
        """sample time"""
        times = torch.zeros((batch,), device=device).float().uniform_(self.sample_range[0],self.sample_range[1])  # [bs]  
        """random noise"""
        noise = torch.randn_like(gt_down)
        noise_level = self.log_snr(times)
        padded_noise_level = self.right_pad_dims_to(img, noise_level)#turn [b]->[b,1,1,1]
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_gt = alpha * gt_down + sigma * noise
        """cat input and noise"""
        feat = torch.cat([feature, noised_gt], dim=1)
        feat = self.transform(feat)#turn b,512,h/4, w/4 to b,256,h/4, w/4
        """conditional input"""
        input_times = self.time_mlp(noise_level)
        loss_decode,_= self.decode_head.forward_train([feat], 
                                                     input_times, 
                                                     img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)
        losses.update(loss_decode)
      
        """aux seg head"""
        # loss_aux = self._auxiliary_head_forward_train([feature], img_metas, gt_semantic_seg)
        # losses.update(loss_aux)
        return losses

    
    