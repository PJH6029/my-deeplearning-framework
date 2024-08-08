import os, gzip, tarfile, pickle
from typing import Literal, Optional, Union, Tuple

try:
    import Image
except ImportError:
    from PIL import Image

import numpy as np
import matplotlib.pyplot as plt


from my_framework.utils import get_file, cache_dir
from my_framework.transforms import Compose, Flatten, ToFloat, Normalize, Transform, Identity
from my_framework.types import NDArray

Data = Union[NDArray, Image.Image]
Target = Union[NDArray, int]

class Dataset:
    def __init__(self, train: bool = True, transform: Optional[Transform] = None, target_transform: Optional[Transform] = None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform # label transform
        if self.transform is None:
            self.transform = Identity()
        if self.target_transform is None:
            self.target_transform = Identity()
            
        self.data = None
        self.label = None
        self.prepare()
        
    def __getitem__(self, index: int) -> tuple[Data, Target]:
        assert np.isscalar(index)
        
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), self.target_transform(self.label[index])
    
    def __len__(self) -> int:
        return len(self.data)
    
    def prepare(self):
        pass
    

# =============================================================================
# Spiral Dataset
# =============================================================================
def get_spiral(train: bool = True) -> tuple[NDArray, NDArray]:
    seed = 1984 if train else 2020 # TODO why?
    np.random.seed(seed)
    
    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_class * num_data
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=np.int32)
    
    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i
            x[ix] = np.array([radius * np.sin(theta), radius * np.cos(theta)]).flatten()
            t[ix] = j
    indices = np.random.permutation(num_data * num_class)
    x = x[indices]
    t = t[indices]
    return x, t

class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)
        
# =============================================================================
# Image Dataset
# =============================================================================
class MNIST(Dataset):
    def __init__(
        self,
        train: bool = True,
        transform: Transform = Compose([Flatten(), ToFloat(), Normalize(0., 255.)]),
        target_transform: Optional[Transform] = None
    ):
        super().__init__(train, transform, target_transform)
    
    def prepare(self):
        url = "http://yann.lecun.com/exdb/mnist/"
        train_files = {
            "images": "train-images-idx3-ubyte.gz",
            "labels": "train-labels-idx1-ubyte.gz"
        }
        test_files = {
            "images": "t10k-images-idx3-ubyte.gz",
            "labels": "t10k-labels-idx1-ubyte.gz"
        }
        
        files = train_files if self.train else test_files
        data_path = get_file(url + files["images"])
        label_path = get_file(url + files["labels"])
        
        self.data = self._load_image(data_path)
        self.label = self._load_label(label_path)
        
    def _load_image(self, path: str) -> NDArray:
        with gzip.open(path, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 28, 28)
        return data

    def _load_label(self, path: str) -> NDArray:
        with gzip.open(path, "rb") as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels
    
    def show(self, row: int = 10, col: int = 10):
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                img[r*H:(r+1)*H, c*W:(c+1)*W] = self.data[np.random.randint(0, len(self.data) - 1)].reshape(H, W)
        plt.imshow(img, cmap="gray", interpolation="nearest")
        plt.axis("off")
        plt.show()
    
    @staticmethod
    def labels() -> dict[int, str]:
        return {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
            5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        }
    
    
class CIFAR10(Dataset):
    def __init__(
        self,
        train: bool = True,
        transform: Transform = Compose([ToFloat(), Normalize(0.5, 0.5)]),
        target_transform: Optional[Transform] = None
    ):
        super().__init__(train, transform, target_transform)
    
    def prepare(self):
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        self.data, self.label = _load_cache_npz(url, self.train)
        if self.data is not None:
            # already cached
            return
        
        filepath = get_file(url)
        if self.train:
            self.data = np.empty((50000, 3 * 32 * 32))
            self.label = np.empty(50000, dtype=np.int32)
            for i in range(5):
                self.data[i * 10000:(i + 1) * 10000] = self._load_image(filepath, i + 1, "train")
                self.label[i * 10000:(i + 1) * 10000] = self._load_label(filepath, i + 1, "train")
        else:
            self.data = self._load_image(filepath, filepath, 5, "test")
            self.label = self._load_label(filepath, 5, "test")
        self.data = self.data.reshape(-1, 3, 32, 32)
        _save_cache_npz(self.data, self.label, url, self.train)
    
    def _load_image(self, filepath: str, idx: int, data_type: Literal["train", "test"]) -> NDArray:
        assert data_type in ["train", "test"]
        with tarfile.open(filepath, "r:gz") as tar:
            for item in tar.getmembers():
                if (f"data_batch_{idx}" in item.name and data_type == "train") or \
                    (f"test_batch" in item.name and data_type == "test"):
                    data_dict = pickle.load(tar.extractfile(item), encoding="bytes")
                    data = data_dict[b"data"]
                    return data
    
    def _load_label(self, filepath: str, idx: int, data_type: Literal["train", "test"]) -> NDArray:
        assert data_type in ["train", "test"]
        with tarfile.open(filepath, "r:gz") as tar:
            for item in tar.getmembers():
                if (f"data_batch_{idx}" in item.name and data_type == "train") or \
                    (f"test_batch" in item.name and data_type == "test"):
                    data_dict = pickle.load(tar.extractfile(item), encoding="bytes")
                    labels = data_dict[b"labels"]
                    return np.array(labels)
    
    def show(self, row: int = 10, col: int = 10):
        H, W = 32, 32
        img = np.zeros((H * row, W * col, 3))
        for r in range(row):
            for c in range(col):
                img[r*H:(r+1)*H, c*W:(c+1)*W] = self.data[np.random.randint(0, len(self.data) - 1)].reshape(3, H, W).transpose(1, 2, 0) / 255.
        plt.imshow(img, interpolation="nearest")
        plt.axis("off")
        plt.show()
    
    @staticmethod
    def labels() -> dict[int, str]:
        return {
            0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
            5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck',
        }

class CIFAR100(CIFAR10):
    def __init__(
        self,
        train: bool = True,
        transform: Transform = Compose([ToFloat(), Normalize(0.5, 0.5)]),
        target_transform: Optional[Transform] = None,
        label_type: Literal["fine", "coarse"] = "fine",
    ):
        assert label_type in ["fine", "coarse"]
        self.label_type = label_type
        super().__init__(train, transform, target_transform)
    
    def prepare(self):
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        self.data, self.label = _load_cache_npz(url, self.train)
        if self.data is not None:
            return
        
        filepath = get_file(url)
        if self.train:
            self.data = self._load_image(filepath, "train")
            self.label = self._load_label(filepath, "train")
        else:
            self.data = self._load_image(filepath, "test")
            self.label = self._load_label(filepath, "test")
        self.data = self.data.reshape(-1, 3, 32, 32)
        _save_cache_npz(self.data, self.label, url, self.train)
    
    def _load_image(self, filepath: str, data_type: Literal["train", "test"]) -> NDArray:
        with tarfile.open(filepath, "r:gz") as tar:
            for item in tar.getmembers():
                if data_type in item.name:
                    data_dict = pickle.load(tar.extractfile(item), encoding="bytes")
                    return data_dict[b"data"]
    
    def _load_label(self, filepath: str, data_type: Literal["train", "test"]) -> NDArray:
        with tarfile.open(filepath, "r:gz") as tar:
            for item in tar.getmembers():
                if data_type in item.name:
                    data_dict = pickle.load(tar.extractfile(item), encoding="bytes")
                    if self.label_type == "fine":
                        labels = data_dict[b"fine_labels"]
                    else:
                        labels = data_dict[b"coarse_labels"]
                    return np.array(labels)
    
    @staticmethod
    def labels(label_type: Literal["fine", "coarse"]) -> dict[int, str]:
        coarse_labels = dict(enumerate(['aquatic mammals','fish','flowers','food containers','fruit and vegetables','household electrical device','household furniture','insects','large carnivores','large man-made outdoor things','large natural outdoor scenes','large omnivores and herbivores','medium-sized mammals','non-insect invertebrates','people','reptiles','small mammals','trees','vehicles 1','vehicles 2']))
        fine_labels = {}
        return fine_labels if label_type == "fine" else coarse_labels

class ImageNet(Dataset):
    def __init__(self, train: bool = True, transform: Transform | None = None, target_transform: Transform | None = None):
        super().__init__(train, transform, target_transform)
        raise NotImplementedError("ImageNet dataset is not implemented yet")
    
    @staticmethod
    def labels() -> dict[int, str]:
        url = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
        path = get_file(url)
        with open(path, "r") as f:
            labels = eval(f.read()) # TODO: eval is dangerous
        return labels

def _load_cache_npz(url: str, train: bool) -> tuple[Union[NDArray, None], Union[NDArray, None]]:
    filename = url[url.rfind("/") + 1:]
    prefix = ".train.npz" if train else ".test.npz"
    filepath = os.path.join(cache_dir, filename + prefix)
    if not os.path.exists(filepath):
        return None, None
    loaded = np.load(filepath)
    return loaded["data"], loaded["label"]

def _save_cache_npz(data: NDArray, label: NDArray, url: str, train: bool) -> str:
    filename = url[url.rfind("/") + 1:]
    prefix = ".train.npz" if train else ".test.npz"
    filepath = os.path.join(cache_dir, filename + prefix)

    if os.path.exists(filepath):
        return
    
    print(f"Saving cache to {filename + prefix}")
    try:
        np.savez_compressed(filepath, data=data, label=label)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise
    print("Done")
    return filepath