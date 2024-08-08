from typing import Union, Optional, Literal
import numpy as np
from numpy.typing import DTypeLike
try:
    import Image
except ImportError:
    from PIL import Image
from my_framework.utils import pairify
from my_framework.types import NDArray

class Transform:
    def __call__(self, img: Union[Image.Image, NDArray]) -> Union[Image.Image, NDArray]:
        raise NotImplementedError()

class Compose(Transform):
    def __init__(self, transforms: list[Transform]=[]):
        self.transforms = transforms
        
    def __call__(self, img: Image.Image) -> Image.Image:
        if not self.transforms:
            return img
        for t in self.transforms:
            img = t(img)
        return img

class Identity(Transform):
    def __call__(self, img: Union[Image.Image, NDArray]) -> Union[Image.Image, NDArray]:
        return img

class Convert(Transform):
    def __init__(self, mode: str):
        self.mode = mode
        
    def __call__(self, img: Image.Image) -> Image.Image:
        if self.mode == "BGR":
            img = img.convert("RGB")
            r, g, b = img.split()
            img = Image.merge("RGB", (b, g, r))
            return img
        else:
            return img.convert(self.mode)


class Resize(Transform):
    def __init__(self, size: Union[int, tuple[int, int]], interpolation: int = Image.BILINEAR):
        self.size = pairify(size)
        self.interpolation = interpolation
    
    def __call__(self, img: Image.Image) -> Image.Image:
        return img.resize(self.size, self.interpolation)


class CenterCrop(Transform):
    def __init__(self, size: Union[int, tuple[int, int]]):
        self.size = pairify(size)
        
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        th, tw = self.size
        left = (w - tw) // 2
        right = w - ((w - tw) // 2 + (w - tw) % 2)
        up = (h - th) // 2
        bottom = h - ((h - th) // 2 + (h - th) % 2)
        return img.crop((left, up, right, bottom))
    

class ToArray(Transform):
    def __init__(self, dtype: DTypeLike = np.float32):
        self.dtype = dtype
    
    def __call__(self, img: Union[Image.Image, NDArray]) -> NDArray:
        if isinstance(img, NDArray):
            return img
        if isinstance(img, Image.Image):
            img = np.asarray(img)
            img = img.transpose(2, 0, 1)
            img = img.astype(self.dtype)
            return img
        else:
            raise TypeError(f"Unsupported type: {type(img)}")


class ToPIL(Transform):
    def __call__(self, array: NDArray) -> Image.Image:
        if array.ndim == 3:
            array = array.transpose(1, 2, 0)
        return Image.fromarray(array)
    

class Normalize(Transform):
    def __init__(
        self, 
        mean: Union[list[float | int], tuple[float | int], float | int] = 0,
        std: Union[list[float | int], tuple[float | int], float | int] = 1,
    ):
        self.mean = mean
        self.std = std
        
    def __call__(self, array: NDArray) -> NDArray:
        mean, std = self.mean, self.std
        
        if not np.isscalar(mean):
            mshape = [1] * array.ndim
            mshape[0] = len(array) if len(mean) == 1 else len(mean)
            mean = np.array(mean, dtype=array.dtype).reshape(*mshape)
        if not np.isscalar(std):
            rshape = [1] * array.ndim
            rshape[0] = len(array) if len(std) == 1 else len(std)
            std = np.array(std, dtype=array.dtype).reshape(*rshape)
        return (array - mean) / std


class Flatten(Transform):
    def __call__(self, array: NDArray) -> NDArray:
        return array.flatten()
    

class AsType(Transform):
    def __init__(self, dtype: DTypeLike = np.float32):
        self.dtype = dtype
        
    def __call__(self, array: NDArray) -> NDArray:
        return array.astype(self.dtype)
    

ToFloat = AsType

class ToInt(AsType):
    def __init__(self, dtype: DTypeLike = np.int32):
        super().__init__(dtype)
