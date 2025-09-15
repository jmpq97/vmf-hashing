import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data_loader import load_data
from models.resnet import resnet34
from models.vmf_loss import VMFLoss, vMF_centroid_loss

import torch.optim as optim
import torch.nn.functional as Fn

X_train, X_valid, X_test, y_train, y_valid, y_test, num_classes, input_size = load_data(mode="train")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_resnet = resnet34(input_size=input_size,num_classes=num_classes,nbits=16)

model_resnet.to(device)

optimizer = optim.Adam(model_resnet.parameters(), lr=0.0005)

train_label = torch.from_numpy(y_train).squeeze().long().to(device)
train_data = torch.from_numpy(X_train).float().to(device)

val_label = torch.from_numpy(y_valid).squeeze().long().to(device)
val_data = torch.from_numpy(X_valid).float().to(device)

assert len(train_data) == len(train_label), "Mismatched lengths"


best_loss = float('inf')

for i in range(200):
    optimizer.zero_grad()

    outputs, embeddings = model_resnet(train_data)
    embeddings = embeddings/embeddings.norm(dim=1).unsqueeze(dim=1)
    train_loss = vMF_centroid_loss(embeddings, train_label, amp=2, amp2=0.01)

    train_loss.backward()
    optimizer.step()
    print("Epoch:", i+1, 'Training loss: ' + str(train_loss.item()))

    with torch.no_grad():
        _, embeddings_val = model_resnet(val_data)
        embeddings_val = embeddings_val/embeddings_val.norm(dim=1).unsqueeze(dim=1)
        val_loss = vMF_centroid_loss(embeddings_val, val_label)
        print("Epoch:", i+1, 'Validation loss: ' + str(val_loss.item()))

    if val_loss.item() < best_loss:
        best_loss = val_loss.item()
        torch.save(model_resnet.state_dict(), 'best_hashnet_model.pth')
        print("New best val loss:", best_loss, "at epoch", i+1)

model_resnet.load_state_dict(torch.load('best_hashnet_model.pth'))
model_resnet.to(device)
model_resnet.eval()

test_label = torch.from_numpy(y_test).squeeze().long().to(device)
test_data = torch.from_numpy(X_test).float().to(device)

with torch.no_grad():
    outputs, embeddings_test = model_resnet(test_data)
    embeddings_test = embeddings_test / embeddings_test.norm(dim=1).unsqueeze(dim=1)
    test_loss = vMF_centroid_loss(embeddings_test, test_label)
    print("Final Test Loss:", test_loss.item())