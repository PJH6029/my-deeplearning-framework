if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import my_framework.datasets
from my_framework.models import MLP
from my_framework import optimizers
import my_framework.functions as F

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

trainset = my_framework.datasets.Spiral()
model = MLP([hidden_size, 3])
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(trainset)
max_iter = data_size // batch_size

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0.0
    
    for i in range(max_iter):
        batch_indices = index[i * batch_size:(i + 1) * batch_size]
        batch = [trainset[i] for i in batch_indices]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])
        
        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        
        model.cleargrads()
        loss.backward()
        optimizer.update()
        
        sum_loss += float(loss.data) * len(batch_t)
    
    avg_loss = sum_loss / data_size
    print(f"epoch: {epoch+1}, loss: {avg_loss:.2f}")