#!/usr/bin/env python
# MIT License
#
# Copyright (c) 2025 Your Name
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

# Use SM_CHANNEL_TRAINING environment variable (for SageMaker) or default to a local directory.
data_dir = os.environ.get("SM_CHANNEL_TRAINING", "image_subset")
annotations_file = os.path.join(data_dir, "train", "annotations.json")
images_dir = os.path.join(data_dir, "train", "images")
print("Using annotations file:", annotations_file)
print("Using images directory:", images_dir)

class LemonDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotation_file, transforms=None):
        self.images_dir = images_dir
        self.coco = COCO(annotation_file)
        # Filter out images that have no annotations.
        self.ids = [img_id for img_id in sorted(self.coco.imgs.keys()) 
                    if len(self.coco.getAnnIds(imgIds=[img_id])) > 0]
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        if file_name.startswith("images/"):
            file_name = file_name[len("images/"):]
        img_path = os.path.join(self.images_dir, file_name)
        # Open the image.
        img = Image.open(img_path).convert("RGB")
        boxes = []
        masks = []
        labels = []
        areas = []
        iscrowd = []
        for ann in anns:
            xmin, ymin, w, h = ann["bbox"]
            boxes.append([xmin, ymin, xmin + w, ymin + h])
            rle = self.coco.annToRLE(ann)
            mask = maskUtils.decode(rle)
            masks.append(mask)
            labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # If there are no masks (shouldn't happen because we filtered), return an empty tensor.
        if masks:
            masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
        else:
            masks = torch.empty((0, img.height, img.width), dtype=torch.uint8)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd
        }
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.ids)

def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        # Additional augmentations can be added here.
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Create training and test datasets.
    dataset = LemonDataset(images_dir, annotations_file, transforms=get_transform(train=True))
    dataset_test = LemonDataset(images_dir, annotations_file, transforms=get_transform(train=False))
    # Create a random train/test split.
    indices = torch.randperm(len(dataset)).tolist()
    split = int(0.9 * len(indices))
    dataset = torch.utils.data.Subset(dataset, indices[:split])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[split:])
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    
    # Load a pre-trained Mask R-CNN model with default weights.
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    # Replace the prediction heads for our 10 classes.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 10)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 10)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    num_epochs = 100
    early_stop_threshold = 0.05  # Stop training if loss falls below this value
    
    for epoch in range(num_epochs):
        model.train()
        for i, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch {epoch}, Iteration {i}, Loss: {loss_value:.05f}")
                
                # Early stopping condition: if loss is below threshold, stop training.
                if loss_value < early_stop_threshold:
                    print(f"Loss {loss_value:.4f} is below threshold {early_stop_threshold}. Stopping early.")
                    torch.save(model.state_dict(), os.path.join(os.environ.get("SM_MODEL_DIR", "./model"), "maskrcnn_lemon.pth"))
                    return
        
        lr_scheduler.step()
        print(f"Epoch {epoch} complete.")
    
    # Save the trained model weights.
    model_dir = os.environ.get("SM_MODEL_DIR", "./model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "maskrcnn_lemon.pth")
    torch.save(model.state_dict(), model_path)
    print("Training complete. Model saved as", model_path)

if __name__ == "__main__":
    main()
