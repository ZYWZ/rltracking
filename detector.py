import torchvision
import torch
import cv2
import numpy as np
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

"""
    model does not perform well, use online faster rcnn instead!
    
"""
if __name__ == "__main__":
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    model.load_state_dict(torch.load('models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'))
    print(model.eval())

    image = cv2.imread('datasets/PETS09/View_001/frame_0000.jpg', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0

    img = torchvision.transforms.ToTensor()(image).unsqueeze_(0)
    # img = torch.from_numpy(image).permute(2, 0, 1)
    # print(img.shape)
    output = model(img)
    classes = output[0]['labels'].detach().numpy()
    print(classes)
    ind = np.where(classes == 1)[0]

    boxes = output[0]['boxes'].detach().numpy()
    people = boxes[ind]
    for person in people:
        cv2.rectangle(image, (person[0], person[1]), (person[2], person[3]), (255, 0, 0), 1)
    cv2.imshow("Image", image)
    cv2.waitKey(-1)