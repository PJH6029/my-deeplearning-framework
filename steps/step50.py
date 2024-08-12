if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import my_framework.datasets
from my_framework.models import MLP
from my_framework import optimizers, DataLoader
from my_framework.datasets import Spiral
import my_framework.functions as F

batch_size = 10
max_epoch = 300
hidden_size = 10
lr = 1.0

train_set = Spiral(train=True)
test_set = Spiral(train=False)

train_loader = DataLoader(train_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []
for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0
    
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)
    
    print(f"epoch: {epoch+1}")
    print(f"train loss: {sum_loss / len(train_set):.4f}, accuracy: {sum_acc / len(train_set):.4f}")
    train_losses.append(sum_loss / len(train_set))
    train_accuracies.append(sum_acc / len(train_set))
    
    
    sum_loss, sum_acc = 0, 0
    with my_framework.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
    print(f"test loss: {sum_loss / len(test_set):.4f}, accuracy: {sum_acc / len(test_set):.4f}")
    test_losses.append(sum_loss / len(test_set))
    test_accuracies.append(sum_acc / len(test_set))
    
import matplotlib.pyplot as plt

_, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(12, 6))
loss_ax.plot(train_losses, label="train")
loss_ax.plot(test_losses, label="test")
loss_ax.legend()

acc_ax.plot(train_accuracies, label="train")
acc_ax.plot(test_accuracies, label="test")
acc_ax.legend()

plt.show()