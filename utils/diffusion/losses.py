# This code is based on https://github.com/openai/guided-diffusion
"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np
import torch as th


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

def geodesic_distance(R_pred, R_gt):
    """
    Compute geodesic loss between predicted and ground truth rotation matrices.
    
    Args:
        R_pred (torch.Tensor): Predicted rotation matrices of shape (batch_size, 3, 3).
        R_gt (torch.Tensor): Ground truth rotation matrices of shape (batch_size, 3, 3).

    Returns:
        torch.Tensor: Geodesic loss for each sample in the batch.
    """
    # Ensure the matrices are valid rotations by using orthogonalization (optional)
    # R_gt = th.nan_to_num(R_gt)
    # U, _, Vt = th.linalg.svd(R_pred)
    # Q = th.matmul(U, Vt) 
    #R_pred_orth = th.linalg.qr(R_pred)[0]
    
    # Compute the relative rotation matrix
    R_rel = th.matmul(R_pred.transpose(-1, -2), R_gt)
    
    # Compute the trace of the relative rotation matrix
    trace = th.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1)
    
    # Compute the geodesic loss (angle in radians)
    epsilon = 1e-6
    theta = th.arccos(th.clamp((trace - 1) / 2, -1+epsilon, 1-epsilon))
    
    return theta[..., None]
