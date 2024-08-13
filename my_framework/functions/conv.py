import numpy as np
from typing import Optional, Union

import my_framework
from my_framework import cuda
from my_framework.core import Variable, Function, as_variable
from my_framework import utils
from my_framework.functions import broadcast_to, linear
import my_framework.functions
from my_framework.types import *
from my_framework.types import NDArray

# =============================================================================
# convolution utility
# =============================================================================
def get_conv_outsize(input_size: int, kernel_size: int, stride: int, pad: int) -> int:
    return (input_size + pad * 2 - kernel_size) // stride + 1


# =============================================================================
# im2col / col2im
# =============================================================================
class Im2col(Function):
    def __init__(
        self, 
        kernel_size: Union[int, tuple[int, int]], 
        stride: Union[int, tuple[int, int]], 
        pad: Union[int, tuple[int, int]], 
        to_matrix: bool
    ) -> None:
        super().__init__()
        self.input_shape: Optional[tuple[int, ...]] = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix
    
    def forward(self, x: NDArray) -> NDArray:
        self.input_shape = x.shape
        y = im2col_array(x, self.kernel_size, self.stride, self.pad, self.to_matrix)
        return y

    def backward(self, gy: Variable) -> Variable:
        gx = col2im(gy, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)
        return gx
    
def im2col(
    x: Union[NDArray, Variable],
    kernel_size: Union[int, tuple[int, int]],
    stride: Union[int, tuple[int, int]] = 1,
    pad: Union[int, tuple[int, int]] = 0,
    to_matrix: bool = True
) -> Variable:
    return Im2col(kernel_size, stride, pad, to_matrix)(x)

class Col2im(Function):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        kernel_size: Union[int, tuple[int, int]],
        stride: Union[int, tuple[int, int]],
        pad: Union[int, tuple[int, int]],
        to_matrix: bool
    ):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix
    
    def forward(self, x: NDArray) -> NDArray:
        y = col2im_array(x, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)
        return y

    def backward(self, gy: Variable) -> Variable:
        gx = im2col(gy, self.kernel_size, self.stride, self.pad, self.to_matrix)
        return gx

def col2im(
    x: Union[NDArray, Variable],
    input_shape: tuple[int, ...],
    kernel_size: Union[int, tuple[int, int]],
    stride: Union[int, tuple[int, int]] = 1,
    pad: Union[int, tuple[int, int]] = 0,
    to_matrix: bool = True
) -> Variable:
    return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(x)


def im2col_array(
    img: NDArray,
    kernel_size: Union[int, tuple[int, int]],
    stride: Union[int, tuple[int, int]],
    pad: Union[int, tuple[int, int]],
    to_matrix: bool = True
):
    N, C, H, W = img.shape
    KH, KW = utils.pairify(kernel_size)
    SH, SW = utils.pairify(stride)
    PH, PW = utils.pairify(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)
    
    xp = cuda.get_array_module(img)
    if xp != np:
        col = _im2col_gpu(img, kernel_size, stride, pad)
    else:
        img = np.pad(
            img,
            ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
            mode='constant',
            constant_values=(0,)
        )
        col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)
        
        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KW):
                i_lim = i + SW * OW
                col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]
        
    if to_matrix:
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1)
    return col

def col2im_array(
    col: NDArray,
    img_shape: tuple[int, ...],
    kernel_size: Union[int, tuple[int, int]],
    stride: Union[int, tuple[int, int]],
    pad: Union[int, tuple[int, int]],
    to_matrix: bool = True
):
    N, C, H, W = img_shape
    KH, KW = utils.pairify(kernel_size)
    SH, SW = utils.pairify(stride)
    PH, PW = utils.pairify(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)
    
    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)
    
    xp = cuda.get_array_module(col)
    if xp != np:
        return _col2im_gpu(col, SH, SW, PH, PW, H, W)
    else:
        img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1), dtype=col.dtype)
        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KH):
                i_lim = i + SW * OW
                img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
        return img[:, :, PH:H + PH, PW:W + PW]


def _im2col_gpu(
    img: NDArray,
    kernel_size: Union[int, tuple[int, int]],
    stride: Union[int, tuple[int, int]],
    pad: Union[int, tuple[int, int]]
) -> NDArray:
    """im2col function for GPU.
    This code is ported from Chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    
    N, C, H, W = img.shape
    KH, KW = utils.pairify(kernel_size)
    SH, SW = utils.pairify(stride)
    PH, PW = utils.pairify(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)
    
    dy, dx = 1, 1
    col = cuda.cupy.empty((N, C, KH, KW, OH, OW), dtype=img.dtype)

    cuda.cupy.ElementwiseKernel(
        'raw T img, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dy, int32 dx',
        'T col',
        '''
           int c0 = i / (kh * kw * out_h * out_w);
           int ky = i / (kw * out_h * out_w) % kh;
           int kx = i / (out_h * out_w) % kw;
           int out_y = i / out_w % out_h;
           int out_x = i % out_w;
           int in_y = ky * dy + out_y * sy - ph;
           int in_x = kx * dx + out_x * sx - pw;
           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
             col = img[in_x + w * (in_y + h * c0)];
           } else {
             col = 0;
           }
        ''',
        'im2col')(img.reduced_view(), H, W, OH, OW, KH, KW, SH, SW, PH, PW, dy, dx, col)

    return col


def _col2im_gpu(
    col: NDArray,
    SY: int,
    SX: int,
    PH: int,
    PW: int,
    H: int,
    W: int
):
    """col2im function for GPU.
    This code is ported from Chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    
    N, C, KH, KW, OH, OW = col.shape
    dx, dy = 1, 1
    img = cuda.cupy.empty((N, C, H, W), dtype=col.dtype)
    
    cuda.cupy.ElementwiseKernel(
        'raw T col, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dx, int32 dy',
        'T img',
        '''
           int c0 = i / (h * w);
           int y  = i / w % h;
           int x  = i % w;
           T val = 0;
           for (int ky = 0; ky < kh; ++ky) {
             int out_y = (y + ph - ky * dy);
             if (0 > out_y || out_y >= out_h * sy) continue;
             if (out_y % sy != 0) continue;
             out_y /= sy;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + pw - kx * dx);
               if (0 > out_x || out_x >= out_w * sx) continue;
               if (out_x % sx != 0) continue;
               out_x /= sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        ''',
        'col2im')(col.reduced_view(),
                    H, W, OH, OW, KH, KW, SY, SX, PH, PW, dx, dy, img)
    return img

# =============================================================================
# convolution
# =============================================================================
def conv2d_simple(
    x: Union[NDArray, Variable],
    W: Union[NDArray, Variable],
    b: Optional[Union[NDArray, Variable]] = None,
    stride: Union[int, tuple[int, int]] = 1,
    pad: Union[int, tuple[int, int]] = 0
):
    x, Weight = as_variable(x), as_variable(W)
    
    N, C, H, W = x.shape
    OC, C, KH, KW = Weight.shape
    SH, SW = utils.pairify(stride)
    PH, PW = utils.pairify(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)
    
    col = im2col(x, (KH, KW), stride, pad, to_matrix=True)
    Weight = Weight.reshape(OC, -1).transpose()
    t = linear(col, Weight, b)
    y = t.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)
    return y


class Conv2d(Function):
    def __init__(self, stride: Union[int, tuple[int, int]] = 1, pad: Union[int, tuple[int, int]] = 0) -> None:
        self.stride = stride
        self.pad = pad
    
    def forward(self, x: NDArray, W: NDArray, b: Optional[NDArray] = None) -> NDArray:
        xp = cuda.get_array_module(x)
        
        KH, KW = W.shape[2:]
        col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=False)
        
        y = xp.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            y += b
        y = xp.rollaxis(y, 3, 1)
        return y
    
    def backward(self, gy: Variable) -> tuple[Variable, Variable, Optional[Variable]]:
        x, W, b = self.inputs
        
        # gx
        gx = deconv2d(gy, W, b=None, stride=self.stride, pad=self.pad, outsize=(x.shape[2], x.shape[3]))
        
        # gW
        gW = Conv2dGradW(self)(x, gy)
        
        # gb
        gb = None
        if b is not None and b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb

def conv2d(
    x: Union[NDArray, Variable],
    W: Union[NDArray, Variable],
    b: Optional[Union[NDArray, Variable]] = None,
    stride: Union[int, tuple[int, int]] = 1,
    pad: Union[int, tuple[int, int]] = 0
):
    return Conv2d(stride, pad)(x, W, b)


class Deconv2d(Function):
    def __init__(
        self,
        stride: Union[int, tuple[int, int]] = 1,
        pad: Union[int, tuple[int, int]] = 0,
        outsize: Optional[tuple[int, int]] = None
    ):
        super().__init__()
        self.stride = utils.pairify(stride)
        self.pad = utils.pairify(pad)
        self.outsize = outsize
    
    def forward(
        self,
        x: NDArray,
        W: NDArray,
        b: Optional[NDArray] = None
    ):
        xp = cuda.get_array_module(x)
        
        Weight = W
        SH, SW = self.stride
        PH, PW = self.pad
        C, OC, KH, KW = Weight.shape
        N, C, H, W = x.shape
        if self.outsize is None:
            OH = get_conv_outsize(H, KH, SH, PH)
            OW = get_conv_outsize(W, KW, SW, PW)
        else:
            OH, OW = utils.pairify(self.outsize)
        
        img_shape = (N, OC, OH, OW)
        
        gcol = xp.tensordot(Weight, x, (0, 1))
        gcol = xp.rollaxis(gcol, 3)
        y = col2im_array(gcol, img_shape, (KH, KW), self.stride, self.pad, to_matrix=False)
        
        if b is not None:
            self.bias = False # TODO
            y += b.reshape(1, -1, 1, 1)
        return y

    def backward(self, gy: Variable) -> tuple[Variable, Variable, Optional[Variable]]:
        x, W, b = self.inputs
        
        # gx
        gx = conv2d(gy, W, b=None, stride=self.stride, pad=self.pad)
        
        # gW
        gW = Conv2dGradW(self)(x, gy)
        
        # gb
        gb = None
        if b is not None and b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb

def deconv2d(
    x: Union[NDArray, Variable],
    W: Union[NDArray, Variable],
    b: Optional[Union[NDArray, Variable]] = None,
    stride: Union[int, tuple[int, int]] = 1,
    pad: Union[int, tuple[int, int]] = 0,
    outsize: Optional[tuple[int, int]] = None
):
    return Deconv2d(stride, pad, outsize)(x, W, b)


class Conv2dGradW(Function):
    def __init__(self, conv2d: Conv2d) -> None:
        W = conv2d.inputs[1]
        KH, KW = W.shape[2:]
        self.kernel_size = (KH, KW)
        self.stride = conv2d.stride
        self.pad = conv2d.pad
    
    def forward(self, x: NDArray, gy: NDArray) -> NDArray:
        xp = cuda.get_array_module(x)
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        gW = xp.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))
        return gW
    
    def backward(self, gy: NDArray) -> tuple[NDArray, NDArray]:
        x, gy = self.inputs
        gW, = self.outputs
        
        XH, XW = x.shape[2:]
        gx = deconv2d(gy, gW, b=None, stride=self.stride, pad=self.pad, outsize=(XH, XW))
        ggy = conv2d(x, gW, b=None, stride=self.stride, pad=self.pad)
        return gx, ggy

