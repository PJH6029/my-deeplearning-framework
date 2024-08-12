import math
import random
import numpy as np

from my_framework.datasets import Dataset
from my_framework import cuda

class DataLoader:
    def __init__(
        self, 
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        gpu: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)
        self.index = None
        
        self.gpu = gpu
        
        self.reset()
        
    def reset(self) -> None:
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration
        
        i, batch_size = self.iteration, self.batch_size
        batch_indices = self.index[i * batch_size:(i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        
        xp = cuda.cupy if self.gpu else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])
        
        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()
    
    def to_cpu(self):
        self.gpu = False
    
    def to_gpu(self):
        self.gpu = True
