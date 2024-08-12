if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import my_framework.datasets
from my_framework.models import MLP
from my_framework import optimizers, DataLoader
import my_framework.datasets
import my_framework.functions as F

max_epoch = 3
batch_size = 100

train_set = my_framework.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
model = MLP((100, 10), activation=F.relu)
optimizer = optimizers.Adam().setup(model)

if os.path.exists("my_mlp.npz"):
    model.load_params("my_mlp.npz")
    print("loaded model parameters")
    
for epoch in range(max_epoch):
    sum_loss = 0
    
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)
    
    print(f"epoch: {epoch+1}, loss: {sum_loss / len(train_set)}")

model.save_params("my_mlp.npz")