from torch import nn


class DynamicMaskAffineFusion(nn.Module):
    """Generate instance masks from shared mask features and dynamic kernels."""

    def __init__(self, num_kernels, norm):
        super().__init__()

        self.affine_scale = nn.Linear(num_kernels, num_kernels, bias=True)
        self.affine_bias = nn.Linear(num_kernels, 1, bias=True)
        self.norm = norm

    def forward(self, mask_features, kernel_features):
        """
        Args:
            mask_features: shared mask feature map with shape (1, C, H, W).
            kernel_features: dynamic mask kernels with shape (N, C).

        Returns:
            Instance mask logits with shape (1, N, H, W).
        """
        kernel_w = self.affine_scale(kernel_features)
        kernel_b = self.affine_bias(kernel_features)
        bs, c, h, w = mask_features.shape
        x = mask_features.view((bs, c, -1))
        if self.norm:
            x_mean = x.mean(2, keepdim=True)
            x_centered = x - x_mean
            x_std_rev = ((x_centered * x_centered).mean(2, keepdim=True) + 1e-10).rsqrt()
            x_norm = x_centered * x_std_rev
        else:
            x_norm = x

        return (kernel_w.matmul(x_norm) + kernel_b).view((bs, -1, h, w))
