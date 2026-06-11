import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from timm.models.layers import to_2tuple
from timm.models.layers import DropPath
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class ALOFTE(nn.Module):
    # 对整个lf特征进行统计量扰动
    def __init__(self, dim, h=14, w=8,
                 mask_radio=0.1, mask_alpha=0.5,
                 noise_mode=1,
                 uncertainty_model=2, perturb_prob=0.5,
                 uncertainty_factor=1.0,
                 noise_layer_flag=0, gauss_or_uniform=0, ):
        super().__init__()
        self.w = w
        self.h = h

        self.mask_radio = mask_radio

        self.noise_mode = noise_mode
        self.noise_layer_flag = noise_layer_flag

        self.alpha = mask_alpha

        self.eps = 1e-6
        self.factor = uncertainty_factor
        self.uncertainty_model = uncertainty_model
        self.p = perturb_prob
        self.gauss_or_uniform = gauss_or_uniform

    def _reparameterize(self, mu, std, epsilon_norm):
        # epsilon = torch.randn_like(std) * self.factor
        epsilon = epsilon_norm * self.factor
        mu_t = mu + epsilon * std
        return mu_t

    def spectrum_noise(self, img, ratio=1.0, noise_mode=1,
                       uncertainty_model=2, gauss_or_uniform=0):
        """Input image size: ndarray of [H, W, C]"""
        """noise_mode: 1 amplitude; 2: phase 3:both"""
        """uncertainty_model: 1 batch-wise modeling 2: channel-wise modeling 3:token-wise modeling"""
        random_val = random.random()
        if random_val > self.p:
            return img
        batch_size, h, w, c = img.shape

        # img_abs_ = img_abs.clone()
        img_abs_ = img.clone()
        # print("img_abs", img_abs_)
        if noise_mode != 0:
            if uncertainty_model != 0:
                if uncertainty_model == 1:
                    # batch level modeling
                    miu = torch.mean(img_abs_, dim=(1, 2), keepdim=True)
                    var = torch.mean((img_abs_ - miu) ** 2, dim=(1, 2), keepdim=True) + self.eps

                    # print("var", var)
                    sig = (var + self.eps).sqrt()  # Bx1x1xC

                    var_of_miu = torch.var(miu, dim=0, keepdim=True)
                    var_of_sig = torch.var(sig, dim=0, keepdim=True)
                    sig_of_miu = (var_of_miu + self.eps).sqrt().repeat(miu.shape[0], 1, 1, 1)
                    sig_of_sig = (var_of_sig + self.eps).sqrt().repeat(miu.shape[0], 1, 1, 1)  # Bx1x1xC

                    if gauss_or_uniform == 0:
                        epsilon_norm_miu = torch.randn_like(sig_of_miu)  # N(0,1)
                        epsilon_norm_sig = torch.randn_like(sig_of_sig)

                        miu_mean = miu
                        sig_mean = sig

                        beta = self._reparameterize(mu=miu_mean, std=sig_of_miu, epsilon_norm=epsilon_norm_miu)
                        gamma = self._reparameterize(mu=sig_mean, std=sig_of_sig, epsilon_norm=epsilon_norm_sig)
                    elif gauss_or_uniform == 1:
                        epsilon_norm_miu = torch.rand_like(sig_of_miu) * 2 - 1.  # U(-1,1)
                        epsilon_norm_sig = torch.rand_like(sig_of_sig) * 2 - 1.
                        beta = self._reparameterize(mu=miu, std=sig_of_miu, epsilon_norm=epsilon_norm_miu)
                        gamma = self._reparameterize(mu=sig, std=sig_of_sig, epsilon_norm=epsilon_norm_sig)
                    else:
                        epsilon_norm_miu = torch.randn_like(sig_of_miu)  # N(0,1)
                        epsilon_norm_sig = torch.randn_like(sig_of_sig)
                        beta = self._reparameterize(mu=miu, std=1., epsilon_norm=epsilon_norm_miu)
                        gamma = self._reparameterize(mu=sig, std=1., epsilon_norm=epsilon_norm_sig)

                    beta = torch.clamp(beta, min=-1.0, max=1.0)
                    gamma = torch.clamp(gamma, min=-1.0, max=1.0)

                    img = gamma * (
                            img - miu) / sig + beta

                elif uncertainty_model == 2:
                    # element level modeling
                    miu_of_elem = torch.mean(img_abs_, dim=0,
                                             keepdim=True)
                    var_of_elem = torch.var(img_abs_, dim=0,
                                            keepdim=True)
                    sig_of_elem = (var_of_elem + self.eps).sqrt()  # 1xHxWxC

                    if gauss_or_uniform == 0:
                        epsilon_sig = torch.randn_like(
                            img)  # BxHxWxC N(0,1)
                        gamma = epsilon_sig * sig_of_elem * self.factor
                    elif gauss_or_uniform == 1:
                        epsilon_sig = torch.rand_like(
                            img) * 2 - 1.  # U(-1,1)
                        gamma = epsilon_sig * sig_of_elem * self.factor
                    else:
                        epsilon_sig = torch.randn_like(
                            img)  # BxHxWxC N(0,1)
                        gamma = epsilon_sig * self.factor

                    img = img + gamma
        return img

    def forward(self, x, spatial_size=None):
        # 只添加特征级的噪声扰动
        B, C, a, b = x.shape

        x = x.view(B, a, b, C)
        x = x.to(torch.float32)

        if self.training:
            if self.noise_mode != 0 and self.noise_layer_flag == 1:
                x = self.spectrum_noise(x, ratio=self.mask_radio, noise_mode=self.noise_mode,
                                        uncertainty_model=self.uncertainty_model,
                                        gauss_or_uniform=self.gauss_or_uniform)
        x = x.reshape(B, C, a, b)
        return x

class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias

class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

def resize_complex_weight(origin_weight, new_h, new_w):
    h, w, num_heads = origin_weight.shape[0:3]  # size, w, c, 2
    origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2)
    new_weight = torch.nn.functional.interpolate(
        origin_weight,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=True
    ).permute(0, 2, 3, 1).reshape(new_h, new_w, num_heads, 2)
    return new_weight

class GlobalFilter(nn.Module):
    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=8, h=48,weight_resize=True,
                 **kwargs):
        super().__init__()
        self.size = h
        self.filter_size = h // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02)     # (48, 25, 4, 2)总共有dim个通道，应该为这dim这通道分别计算逐通道的滤波器权重，而不是只用4个滤波器
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)
        
    def forward(self, x):
        B, C, H, W = x.shape

        x = x.view(B, H, W, C)
        x = x.to(torch.float32)

        routeing = self.reweight(x.mean(dim=(1, 2))).view(B, self.num_filters,
                                                          -1).softmax(dim=1)    # (32, 4, 48)
        x = self.pwconv1(x)   # (32, 48 48 48)
        x = self.act1(x)    
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        if self.weight_resize:
            complex_weights = resize_complex_weight(self.complex_weights, x.shape[1],
                                                    x.shape[2])
            complex_weights = torch.view_as_complex(complex_weights.contiguous())
        else:
            complex_weights = torch.view_as_complex(self.complex_weights)   # (48, 25, 4)
        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)
        if self.weight_resize:
            weight = weight.view(-1, x.shape[1], x.shape[2], self.med_channels)
        else:
            weight = weight.view(-1, self.size, self.filter_size, self.med_channels)   # (32, 48, 25, 48)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        x = self.act2(x)
        x = self.pwconv2(x)   # (32, 48,48,24)
        x = x.reshape(B, C, H, W)
        return x



"""
论文地址：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06713.pdf
论文题目：SMFANet: A Lightweight Self-Modulation Feature Aggregation Network for Efficient Image Super-Resolution（ECCV 2024）
讲解视频：https://www.bilibili.com/video/BV1sxcde4EpX/

  高效近似自注意力 (Efficient Approximation of Self-attention,EASA)
     为了获得低频信息，首先经过缩放因子为8的自适应最大池化操作D，接着传入一个3×3深度可分离卷积层来生成非局部结构信息，
     接着引入了 X 的方差作为空间信息的统计差异，并通过1×1卷积将其与非局部 Xs 合并得到调制特征，
     最后，聚合调制特征与输入特征X得到代表性结构信息 Xl。
     
"""

class EASA(nn.Module):
    def __init__(self, dim=36):
        super(EASA, self).__init__()
        # 定义1x1卷积层，将输入通道数从dim扩展到2*dim
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        # 定义1x1卷积层，保持通道数不变
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        # 定义1x1卷积层，保持通道数不变
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        # 定义深度可分离卷积层（Depth-wise Convolution），保持通道数不变
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        # 定义GELU激活函数
        self.gelu = nn.GELU()
        # 设置下采样因子
        self.down_scale = 8

        # 定义可学习参数alpha，初始化为全1
        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        # 定义可学习参数belt，初始化为全0
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, X):
        _, _, h, w = X.shape  # 获取输入特征图的高度和宽度

        # 对X进行自适应最大池化操作，然后通过深度可分离卷积层
        x_s = self.dw_conv(F.adaptive_max_pool2d(X, (h // self.down_scale, w // self.down_scale)))

        # 计算X的方差 ，作为空间信息的统计差异
        x_v = torch.var(X, dim=(-2, -1), keepdim=True)

        # 计算局部细节估计
        Temp = x_s * self.alpha + x_v * self.belt

        x_l = X * F.interpolate(self.gelu(self.linear_1(Temp)), size=(h, w), mode='nearest')

        # 通过线性层2输出最终结果
        return self.linear_2(x_l)

"""
论文地址：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06713.pdf
论文题目：SMFANet: A Lightweight Self-Modulation Feature Aggregation Network for Efficient Image Super-Resolution（ECCV 2024）
讲解视频：https://www.bilibili.com/video/BV1Wrwwe6EjR/

    局部细节估计 (Local Detail Estimation, LDE)
       为了获得高频成分，首先使用一个3×3深度可分离卷积来从输入特征Y中编码局部信息Yh。
       然后，使用两个带有隐藏GELU激活函数的1×1卷积生成增强的局部特征Yd。
       通过这种方式，LDE分支能够有效地捕捉到图像中的局部细节，这些细节对于提高超分辨率重建的质量至关重要。
"""
class LDE(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()  # 调用父类nn.Module的初始化方法
        hidden_dim = int(dim * growth_rate)  # 计算隐藏层维度，基于输入维度和增长因子
        self.conv_0 = nn.Sequential(  # 定义第一个卷积序列
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),  # 深度可分离卷积，使用分组卷积来减少参数
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)      # 点卷积，用于通道融合
        )
        self.act = nn.GELU()  # 使用GELU激活函数
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)  # 最后一个点卷积，将隐藏层维度还原到输入维度

    def forward(self, x):
        x = self.conv_0(x)  # 应用第一个卷积序列
        x = self.act(x)  # 应用激活函数
        x = self.conv_1(x)  # 应用最后一个点卷积
        return x  # 返回最终输出


#GitHub地址：https://github.com/YOLOonMe/EMA-attention-module
#论文地址：https://arxiv.org/abs/2305.13563v2
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

#### baseline+dynamic global filiter varient-----相比于DGF，在DGF前加了特征级的噪声扰动,第一次num_filiter=32,第二次为8
class BackBoneAug(nn.Module):
    def __init__(self, dim, drop_path=0.,h=48, w=48, norm_layer=nn.BatchNorm2d,):
        super().__init__()
        self.alofte = ALOFTE(dim=dim,h=h, w=w, uncertainty_model=2)
        self.norm1 = norm_layer(dim)
        self.act1 = nn.GELU()
        self.fc1 = nn.Conv2d(dim, dim, 1, 1)
        self.filter = GlobalFilter(dim=dim, num_filters=8, h=h)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.easa = EASA(dim=dim)
        self.lde = LDE(dim=dim)
        self.ema = EMA(channels=dim)

    # def forward(self, input):
    #     x = input
    #     x = x + self.drop_path(self.alofte(x))
    #     x = self.ema(x)
    #     # x1 = self.easa(x)
    #     # x2 = self.lde(x)
    #     # x = x + self.act1(x1 + x2)
    #     x = self.alofte(x)
    #     x = x + self.drop_path(self.filter(self.norm1(x)))

    #     # Drop_path: In residual architecture, drop the current block for randomly seleted samples
    #     return x
    def forward(self, input):
        x = input
        # x = self.ema(x)
        x = self.alofte(x)
        x = x + self.drop_path(self.filter(self.norm1(x)))
        # Drop_path: In residual architecture, drop the current block for randomly seleted samples
        return x