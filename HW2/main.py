#!/usr/bin/env python3
import torch
import pprint
import torchvision
from torchvision import models, transforms
from PIL import Image

image_names = ["candy4.jpeg", "candy5.jpeg"]
images = [Image.open(img) for img in image_names]

resnet = models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img_tensors = [preprocess(img) for img in images]
batch_tensors = [torch.unsqueeze(img_t, 0) for img_t in img_tensors]

resnet.eval()
outputs = [resnet(batch_t) for batch_t in batch_tensors]

for i in outputs:
    _, index = torch.max(i, 1)

indices = [torch.max(out, 1).indices for out in outputs]
print(indices)
print()

with open("imagenet1000.txt", 'r') as f:
    lines = [line.strip() for line in f.readlines()]

    for i, val in enumerate(images):
        print(image_names[i])
        _, indices = torch.sort(outputs[i], descending=True)
        percentage = torch.nn.functional.softmax(outputs[i], dim=1)[0] * 100
        temp = [(lines[idx], percentage[idx].item()) for idx in indices[0][:10]]
        for i in temp:
            print(i)
        print()
