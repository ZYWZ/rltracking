import os.path
from os import walk

import torch
import torch.nn as nn
from torch.optim import Adam
from models.extractor import build_extractor, build_MDNet
from PIL import Image
from torchvision.models import resnet50, resnet152
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

"""
    input: img(t), img(t-1)
    output: feat(t), feat(t-1)
    loss: triplet loss
    
"""

INPUT_PATH_TRAIN = "datasets/2DMOT2015/train"
INPUT_PATH_TEST = "datasets/2DMOT2015/test"

MODEL_PATH = "model_state_dict/state_dict_extractor.pt"

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def plot_loss(losses, lr):
    epochs = range(0, int(len(losses)))
    sample_loss = []
    for i, loss in enumerate(losses):
        sample_loss.append(loss)

    plt.plot(epochs, sample_loss, 'b', label='Extractor training loss')
    plt.title('Training loss of extractor, lr=' + str(lr))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def box_cxcywh_to_xyxy(x):
    frame, label, x_c, y_c, w, h = x
    b = [frame, label, x_c, y_c,
         (x_c + w), (y_c + h)]
    return b


def load_detection_result():
    _, directories, _ = next(walk(INPUT_PATH_TRAIN))
    results = []
    for directory in directories:
        file = os.path.join(INPUT_PATH_TRAIN, directory, "det", "det.txt")
        with open(file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        results.append(content)

    output = dict(zip(directories, results))

    return output

def load_gt():
    _, directories, _ = next(walk(INPUT_PATH_TRAIN))
    results = []
    for directory in directories:
        file = os.path.join(INPUT_PATH_TRAIN, directory, "gt", "gt.txt")
        with open(file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        results.append(content)

    output = dict(zip(directories, results))

    return output


DET_RESULT = load_detection_result()
GT_RESULT = load_gt()


def resize_roi(roi, w, h):
    w = 720
    h = 576
    roi[:, 1] = roi[:, 1] * 1066 / w
    roi[:, 3] = roi[:, 3] * 1066 / w
    roi[:, 2] = roi[:, 2] * 800 / h
    roi[:, 4] = roi[:, 4] * 800 / h
    return roi


def get_detection(source, frame):
    frame = int(frame)
    output = []
    for line in GT_RESULT[source]:
        line = line.split(',')
        if int(line[0]) == frame:
            temp = []
            for i in line[:6]:
                temp.append(float(i))
            temp = box_cxcywh_to_xyxy(temp)
            output.append(temp)
    return output


def get_random_apn(feat1, feat2, det1, det2):
    idx1 = det1[:, 1].flatten()
    idx2 = det2[:, 1].flatten()
    if idx1.shape[0] < 2 and idx2.shape[0] < 2:
        return None, None, None
    a = [i for i in range(idx1.shape[0])]
    rand = np.random.choice(a, 2, replace=False)
    anchor = rand[0]
    negative = rand[1]
    if idx1[anchor] not in idx2:
        return None, None, None
    for i, id in enumerate(idx2):
        if id == idx1[anchor]:
            positive = i
    anchor = feat1[0, anchor, :].unsqueeze(0)
    positive = feat2[0, positive, :].unsqueeze(0)
    negative = feat1[0, negative, :].unsqueeze(0)
    return anchor, positive, negative

def train_one_step(model, device, optimizer, source, frame1, frame2):
    path1 = os.path.join("datasets/2DMOT2015/train", source, "img1", frame1 + ".jpg")
    path2 = os.path.join("datasets/2DMOT2015/train", source, "img1", frame2 + ".jpg")

    img1 = Image.open(path1)
    img2 = Image.open(path2)

    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)

    det1 = torch.FloatTensor(get_detection(source, frame1))
    det2 = torch.FloatTensor(get_detection(source, frame2))

    roi1 = det1[:, 2:]
    roi2 = det2[:, 2:]

    pad = (1, 0)
    roi1 = F.pad(roi1, pad, 'constant', 0).to(device)
    roi2 = F.pad(roi2, pad, 'constant', 0).to(device)

    roi1 = resize_roi(roi1, 720, 576)
    roi2 = resize_roi(roi2, 720, 576)

    feat1 = model(roi1, img1)
    feat2 = model(roi2, img2)

    anchor, positive, negative = get_random_apn(feat1, feat2, det1, det2)
    if anchor is not None:
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        loss = triplet_loss(anchor, positive, negative)
        output = str(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return output

    return None

def get_index(frame):
    result = ""
    digit = len(str(frame))
    max = 6
    for i in range(max - digit):
        result += "0"
    result += str(frame)

    return result


if __name__ == "__main__":
    model = build_extractor().cuda()
    model.eval()
    img = Image.open("datasets/2DMOT2015/train/PETS09-S2L1/img1/000001.jpg")
    img = transform(img).unsqueeze(0).cuda()
    det = torch.FloatTensor([[0, 500, 158, 530, 228],
                               [0, 246, 218, 286, 309],
                               [0, 648, 238, 684, 321]]).cuda()
    det = resize_roi(det, 1, 1)
    output = model(det, img)

    # model = build_extractor()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # print(model.eval())
    # model.train()
    # optimizer = Adam(model.parameters(), lr=0.0001)
    # print("start to train feature extraction backbone")
    #
    # source = "PETS09-S2L1"
    # losses = []
    # # frame1 = "000001"
    # # frame2 = "000002"
    # for i in range(1, 794):
    #     frame1 = get_index(i)
    #     frame2 = get_index(i+1)
    #     loss = train_one_step(model, device, optimizer, source, frame1, frame2)
    #     if loss is not None:
    #         print("step ", i, ", loss: ", loss)
    #         losses.append(loss)
    #
    # print("Saving model...")
    # torch.save(model.state_dict(), MODEL_PATH)
    #
    # plot_loss(losses, optimizer.param_groups[0]['lr'])

    # img1 = Image.open("datasets/2DMOT2015/train/PETS09-S2L1/img1/000001.jpg")
    # img2 = Image.open("datasets/2DMOT2015/train/PETS09-S2L1/img1/000002.jpg")
    #
    # img1 = transform(img1).unsqueeze(0).to(device)
    # img2 = transform(img2).unsqueeze(0).to(device)
    #
    # det1 = torch.FloatTensor([[0, 500, 158, 530, 228],
    #                            [0, 246, 218, 286, 309],
    #                            [0, 648, 238, 684, 321]]).to(device)
    #
    # det2 = torch.FloatTensor([[0, 495, 158, 525, 228],
    #                            [0, 245, 215, 288, 313],
    #                            [0, 637, 245, 670, 321],
    #                            [0, 504, 162, 535, 232]]).to(device)
    # det1 = resize_roi(det1, 1, 1)
    # det2 = resize_roi(det2, 1, 1)
    #
    # feat1 = model(det1, img1)
    # feat2 = model(det2, img2)
    #
    #
    # a = feat1[0, 0, :].unsqueeze(0)
    # p = feat2[0, 0, :].unsqueeze(0)
    # n = feat1[0, 1, :].unsqueeze(0)
    #
    # triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    # output = triplet_loss(a, p, n)
    # optimizer = Adam(model.parameters(), lr=0.0001)
    #
    # optimizer.zero_grad()
    # output.backward()
    # optimizer.step()
