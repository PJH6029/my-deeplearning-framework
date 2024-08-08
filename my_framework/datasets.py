import os, gzip, tarfile, pickle
import numpy as np
import matplotlib.pyplot as plt
from my_framework.utils import get_file, cache_dir
from my_framework.transforms import Compose, Flatten, ToFloat, Normalize

