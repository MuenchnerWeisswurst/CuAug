from typing import Tuple, Union

import torch
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import map_coordinates, gaussian_filter, affine_transform
from torch.utils.dlpack import to_dlpack, from_dlpack


class RandomGamma(object):
    def __init__(self,
                 log_gamma: Tuple[float, float],
                 random_state=None
                 ):
        self.low, self.high = log_gamma
        if not random_state:
            self.random_state = cp.random.RandomState(None)
        else:
            self.random_state = random_state

    def __call__(self, img: cp.ndarray) -> cp.ndarray:
        gamma = (self.high - self.low) * self.random_state.rand(1) + self.low
        return cp.power(img, cp.exp(gamma))


class RandomNoise(object):
    def __init__(self,
                 mean: Tuple[float, float],
                 sigma: Tuple[float, float],
                 random_state=None
                 ):
        self.mean_min, self.mean_max = mean
        self.sigma_min, self.sigma_max = sigma
        if not random_state:
            self.random_state = cp.random.RandomState(None)
        else:
            self.random_state = random_state

    def __call__(self, img: cp.ndarray) -> cp.ndarray:
        mean = (self.mean_max - self.mean_min) * self.random_state.rand(1) + self.mean_min
        sigma = (self.sigma_max - self.sigma_min) * self.random_state.rand(1) + self.sigma_min

        noise_vec = (mean + self.random_state.randn(*img.shape)) * sigma
        return img + cp.stack(3 * [noise_vec])


class RandomGaussianBlur(object):
    def __init__(self,
                 sigma: Tuple[float, float],
                 random_state=None
                 ):
        self.sigma = sigma
        if not random_state:
            self.random_state = cp.random.RandomState(None)
        else:
            self.random_state = random_state

    def __call__(self, img: cp.ndarray) -> cp.ndarray:
        sigma_x, sigma_y, sigma_z = _get_random_inputs(self.sigma, self.random_state)
        return cp.stack([gaussian_filter(img[channel], (sigma_x, sigma_y, sigma_z), mode='constant') for channel in
                         range(img.shape[0])])


class RandomElasticDeformation(object):
    def __init__(self,
                 alpha: Tuple[float, float],
                 sigma: Tuple[float, float],
                 random_state=None
                 ):
        self.alpha = alpha
        self.sigma = sigma
        self.random_state = random_state

        if not self.random_state:
            self.random_state = cp.random.RandomState(None)
        else:
            self.random_state = random_state

    def __call__(self, img: cp.ndarray) -> cp.ndarray:
        # NumPy implementation taken from: https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
        shape = img.shape[1:]
        alpha_x, alpha_y, alpha_z = _get_random_inputs(self.alpha, self.random_state)
        sigma_x, sigma_y, sigma_z = _get_random_inputs(self.sigma, self.random_state)
        dx = gaussian_filter(self.random_state.rand(*shape) * 2 - 1, sigma_x, mode='constant', cval=0) * alpha_x
        dy = gaussian_filter(self.random_state.rand(*shape) * 2 - 1, sigma_y, mode='constant', cval=0) * alpha_y
        dz = gaussian_filter(self.random_state.rand(*shape) * 2 - 1, sigma_z, mode='constant', cval=0) * alpha_z

        x, y, z = cp.meshgrid(
            cp.arange(shape[0]),
            cp.arange(shape[1]),
            cp.arange(shape[2]),
            indexing='ij'
        )
        indices = cp.reshape(x + dx, (-1, 1)), cp.reshape(y + dy, (-1, 1)), cp.reshape(z + dz, (-1, 1))
        indices = cp.stack(indices, axis=1).squeeze().T
        return cp.stack(
            [map_coordinates(img[channel], indices, order=1).reshape(shape) for channel in range(img.shape[0])])


class RandomAffine(object):
    def __init__(self,
                 degrees: Union[int, Tuple[int, int], Tuple[int, int, int, int, int, int]],
                 scales: Union[int, Tuple[int, int], Tuple[int, int, int, int, int, int]],
                 translation: Union[int, Tuple[int, int], Tuple[int, int, int, int, int, int]] = (0, 0),
                 random_state=None
                 ):
        self.scales = scales
        self.degrees = degrees
        self.translation = translation
        if not random_state:
            self.random_state = np.random.RandomState(None)
        else:
            self.random_state = random_state

    def __call__(self, img: cp.ndarray) -> cp.ndarray:
        scales = _get_random_inputs(self.scales, self.random_state)
        degrees = _get_random_inputs(self.degrees, self.random_state)
        translation = _get_random_inputs(self.translation, self.random_state)

        scale_mat = cp.diag(scales + (1,))
        rot_mat = cp.identity(4)
        for i, d in enumerate(degrees):
            rad = np.deg2rad(d)
            c, s = np.cos(rad), np.sin(rad)
            if i == 0:  # x-axis
                rot_mat = rot_mat @ cp.array([
                    [1, 0, 0, 0],
                    [0, c, -s, 0],
                    [0, s, c, 0],
                    [0, 0, 0, 1]
                ])
            elif i == 1:  # y-axis
                rot_mat = rot_mat @ cp.array([
                    [c, 0, s, 0],
                    [0, 1, 0, 0],
                    [-s, 0, c, 0],
                    [0, 0, 0, 1]
                ])
            elif i == 2:  # z-axis
                rot_mat = rot_mat @ cp.array([
                    [c, -s, 0, 0],
                    [s, c, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])

        trans_mat = cp.identity(4)
        trans_mat[:3, 3] = cp.array(translation)

        affine_mat = rot_mat @ scale_mat @ trans_mat

        return cp.stack([affine_transform(img[channel], affine_mat, order=1) for channel in range(img.shape[0])])


class ToNumpy(object):
    def __call__(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.numpy()


class ToTensor(object):
    def __call__(self, arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).contiguous()


class NumpyToCupy(object):
    def __call__(self, arr: np.ndarray) -> cp.ndarray:
        return cp.array(arr)


class CupyToTensor(object):
    def __call__(self, carr: cp.ndarray) -> torch.Tensor:
        return from_dlpack(carr.toDlpack())


class TensorToCupy(object):
    def __call__(self, t: torch.Tensor) -> cp.ndarray:
        return cp.fromDlpack(to_dlpack(t))


class CupyToNumpy(object):
    def __call__(self, carr: cp.ndarray) -> np.ndarray:
        return carr.get()


def _get_random_inputs(value_range, random_state):
    if isinstance(value_range, int) or isinstance(value_range, float):
        return _get_random_inputs((-value_range, value_range), random_state)
    if len(value_range) == 6:
        xl, xh, yl, yh, zl, zh = value_range
        x = (xh - xl) * random_state.rand(1) + xl
        y = (yh - yl) * random_state.rand(1) + yl
        z = (zh - xl) * random_state.rand(1) + zl
        return x[0], y[0], z[0]
    elif len(value_range) == 2:
        l, h = value_range
        x = (h - l) * random_state.rand(1) + l
        y = (h - l) * random_state.rand(1) + l
        z = (h - l) * random_state.rand(1) + l
        return x[0], y[0], z[0]
    elif len(value_range) == 1:
        return _get_random_inputs((-value_range[0], value_range[0]))
    else:
        raise Exception(f"Range has too many/less values either 6 or 2 instead of {len(value_range)}")
