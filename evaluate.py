import logging
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from data_loader import load_data
from models.resnet import resnet34
from models.vmf_loss import VMFLoss, vMF_centroid_loss 

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

X_train, X_valid, X_test, y_train, y_valid, y_test, num_classes, input_size = load_data(mode="eval")

def tensor_to_dataset(tensor, transform=None):
    class TransformTensorDataset(Dataset):
        def __init__(self, tensor, ts=None):
            super(TransformTensorDataset, self).__init__()
            self.tensor = tensor
            self.ts = ts

        def __getitem__(self, index):
            if self.ts is not None:
                return self.ts(self.tensor[index])
            return self.tensor[index]

        def __len__(self):
            return len(self.tensor)

    ttd = TransformTensorDataset(tensor, transform)
    return ttd

class Timer(object):
    def __init__(self):
        self.start = 0
        self.end = 0
        self.total = 0

    def tick(self):
        self.start = time.time()
        return self.start

    def toc(self):
        self.end = time.time()
        self.total = self.end - self.start
        return self.end

    def print_time(self, title):
        print(f'{title} time: {self.total:.4f}s')


def get_hamm_dist(codes, centroids, margin=0, normalize=False):
    with torch.no_grad():
        nbit = centroids.size(1)
        dist = 0.5 * (nbit - torch.matmul(codes.sign(), centroids.sign().t()))

        if normalize:
            dist = dist / nbit

        if margin == 0:
            return dist
        else:
            codes_clone = codes.clone()
            codes_clone[codes_clone.abs() < margin] = 0
            dist_margin = 0.5 * (nbit - torch.matmul(codes_clone.sign(), centroids.sign().t()))
            if normalize:
                dist_margin = dist_margin / nbit
            return dist, dist_margin


def get_codes_and_labels(model, loader):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    vs = []
    ts = []
    for e, (d, t) in enumerate(loader):
        print(f'[{e + 1}/{len(loader)}]', end='\r')
        with torch.no_grad():
            # model forward
            d, t = d.to(device), t.to(device)
            v = model(d)
            if isinstance(v, tuple):
                v = v[0]

            vs.append(v)
            ts.append(t)

    print()
    vs = torch.cat(vs)
    ts = torch.cat(ts)
    return vs, ts

def dataloader(d, bs=256, shuffle=True, workers=-1, drop_last=True):
    if workers < 0:
        workers = 16
    l = DataLoader(d,
                   bs,
                   shuffle,
                   drop_last=drop_last,
                   num_workers=workers)
    return l


def calculate_mAP(db_codes, db_labels,
                  test_codes, test_labels,
                  R, threshold=0.):
    db_codes = db_codes.clone()
    test_codes = test_codes.clone()

    if threshold != 0:
        db_codes[db_codes.abs() < threshold] = 0
        test_codes[test_codes.abs() < threshold] = 0

    db_codes = torch.sign(db_codes)
    test_codes = torch.sign(test_codes)

    db_labels = db_labels.cpu().numpy() #
    test_labels = test_labels.cpu().numpy()

    dist = []
    nbit = db_codes.size(1)

    timer = Timer()
    total_timer = Timer()

    timer.tick()
    total_timer.tick()

    with torch.no_grad():
        db_codes_ttd = tensor_to_dataset(db_codes)
        db_codes_loader = dataloader(db_codes_ttd, 32, False, 0, False)

        for i, db_code in enumerate(db_codes_loader):
            dist.append(0.5 * (nbit - torch.matmul(test_codes, db_code.t())).cpu())
            timer.toc()
            print(f'Distance [{i + 1}/{len(db_codes_loader)}] ({timer.total:.2f}s)', end='\r')

        dist = torch.cat(dist, 1)
        print()


    timer.tick()
    topk_ids = torch.topk(dist, R, dim=1, largest=False)[1].cpu()
    timer.toc()
    print(f'Sorting ({timer.total:.2f}s)')

    timer.tick()
    APx = []
    for i in range(dist.shape[0]):
        label = test_labels[i, :]
        label[label == 0] = -1
        idx = topk_ids[i, :]
        imatch = np.sum(np.equal(db_labels[idx[0: R], :], label), 1) > 0
        rel = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)
        if rel != 0:
            APx.append(np.sum(Px * imatch) / rel)
        else:
            APx.append(0)
        timer.toc()
        print(f'Query [{i + 1}/{dist.shape[0]}] ({timer.total:.2f}s)', end='\r')

    print()
    total_timer.toc()
    logging.info(f'Total time usage for calculating mAP: {total_timer.total:.2f}s')
    return np.mean(np.array(APx)),dist

def sign_dist(inputs, centroids, margin=0):
    n, b1 = inputs.size()
    nclass, b2 = centroids.size()

    assert b1 == b2, 'inputs and centroids must have same number of bit'

    out = inputs.view(n, 1, b1) * centroids.sign().view(1, nclass, b1)
    out = torch.relu(margin - out)

    return out


def calculate_similarity_matrix(centroids):
    nclass = centroids.size(0)
    sim = torch.zeros(nclass, nclass, device=centroids.device)

    for rc in range(nclass):
        for cc in range(nclass):
            sim[rc, cc] = (centroids[rc] == centroids[cc]).float().mean()

    return sim


def first_ten_labels_data(input_data, input_label):
    input_label = input_label.reshape(-1)

    input_label = torch.tensor(input_label)

    mask = input_label < 10

    label_revised = input_label[mask]
    data_revised = input_data[mask]

    label_revised = label_revised.reshape([-1,1])
    return data_revised, label_revised


def compute_mAP_with_testdb(test_hashcode, test_labels,R=10):

    idx = torch.randn((test_hashcode.shape[0]))>0
    test_labels = test_labels.reshape(-1)

    max_label = test_labels.max() + 1
    max_label = max_label.astype('int')
    test_labels = torch.nn.functional.one_hot(torch.tensor(test_labels).long(), num_classes=max_label).numpy()


    db_codes = test_hashcode[idx,:]
    test_codes = test_hashcode[~idx,:]
    db_labels = test_labels[idx,:]
    test_labels = test_labels[~idx,:]

    dist = []
    nbit = db_codes.shape[1]

    mAP = calculate_mAP(db_codes, torch.tensor(db_labels), test_codes, torch.tensor(test_labels),R=10)
    return mAP

test_label = torch.from_numpy(y_test).squeeze().long().to(device)
test_data = torch.from_numpy(X_test).float().to(device)

model_resnet = resnet34(input_size=input_size, num_classes=num_classes, nbits=16)
model_resnet.load_state_dict(torch.load("best_model.pth", map_location=device))
model_resnet.to(device)
model_resnet.eval()

train_label = torch.from_numpy(y_train).squeeze().long().to(device)
train_data = torch.from_numpy(X_train).float().to(device)

test_label = torch.from_numpy(y_test).squeeze().long().to(device)
test_data = torch.from_numpy(X_test).float().to(device)

out = model_resnet(test_data)
output_hash = out[1]/out[1].norm(dim=1).unsqueeze(dim=1)

out = model_resnet(train_data)
db_hash = out[1]/out[1].norm(dim=1).unsqueeze(dim=1)

[mAP, code] = compute_mAP_with_testdb(output_hash.cpu(), test_label.cpu().detach().numpy())

print("Mean Average Precision (mAP):", mAP)