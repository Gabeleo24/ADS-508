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
import argparse
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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iter', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--early_stop_threshold', type=float, default=0.05)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    return parser.parse_args()

# Use SM_CHANNEL_TRAINING environment variable (for SageMaker) or default to a local directory
data_dir = os.environ.get("SM_CHANNEL_TRAINING", "image_subset")
annotations_file = os.path.join(data_dir, "train", "annotations.json")
images_dir = os.path.join(data_dir, "train", "images")
print("Using annotations file:", annotations_file)
print("Using images directory:", images_dir)

class LemonDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotation_file, transforms=None):
        self.images_dir = images_dir
        self.coco = COCO(annotation_file)
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
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([img_id])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)

class ModelEvaluator:
    def __init__(self, model, data_loader, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.predictions = []
        self.ground_truth = []
        
    def collect_predictions(self):
        self.model.eval()
        with torch.no_grad():
            for images, targets in self.data_loader:
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)
                
                for output, target in zip(outputs, targets):
                    mask = output['scores'] > 0.5
                    labels = output['labels'][mask].cpu().numpy()
                    
                    gt_labels = target['labels'].cpu().numpy()
                    
                    if len(labels) > 0:
                        pred_label = np.bincount(labels).argmax()
                    else:
                        pred_label = 0
                        
                    if len(gt_labels) > 0:
                        true_label = np.bincount(gt_labels).argmax()
                    else:
                        true_label = 0
                    
                    self.predictions.append(pred_label)
                    self.ground_truth.append(true_label)
    
    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.ground_truth, self.predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
    def calculate_metrics(self):
        return {
            'accuracy': accuracy_score(self.ground_truth, self.predictions),
            'precision': precision_score(self.ground_truth, self.predictions, average='weighted'),
            'recall': recall_score(self.ground_truth, self.predictions, average='weighted'),
            'f1': f1_score(self.ground_truth, self.predictions, average='weighted')
        }

def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Create datasets
    dataset = LemonDataset(images_dir, annotations_file, transforms=get_transform(train=True))
    dataset_test = LemonDataset(images_dir, annotations_file, transforms=get_transform(train=False))
    
    # Split datasets
    indices = torch.randperm(len(dataset)).tolist()
    split = int(0.9 * len(indices))
    dataset = torch.utils.data.Subset(dataset, indices[:split])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[split:])
    
    # Create data loaders with batch size from args
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    # Initialize model
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    
    # Replace the prediction heads
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 10)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 10)
    model.to(device)

    # Initialize optimizer with learning rate from args
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop with parameters from args
    print("Starting training...")
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        for i, (images, targets) in enumerate(data_loader):
            if i >= args.max_iter:
                break
                
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            epoch_loss += loss_value
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch {epoch}, Iteration {i}, Loss: {loss_value:.5f}")
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch} complete. Average loss: {avg_epoch_loss:.5f}")
        
        # Early stopping check with threshold from args
        if avg_epoch_loss < args.early_stop_threshold:
            print(f"Loss {avg_epoch_loss:.4f} is below threshold {args.early_stop_threshold}. Stopping early.")
            break
        
        lr_scheduler.step()

    print("Training complete. Running evaluation...")
    
    # Evaluate the model
    evaluator = ModelEvaluator(model, data_loader_test, device)
    evaluator.collect_predictions()
    
    # Generate visualizations and metrics
    evaluator.plot_confusion_matrix()
    metrics = evaluator.calculate_metrics()
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save the model to the path specified by SageMaker
    model_dir = os.environ.get('SM_MODEL_DIR', '.')
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

if __name__ == "__main__":
    main()