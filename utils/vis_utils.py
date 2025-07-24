import os
from typing import List

import torchvision.transforms as tvtf
import supervision as sv
from torchvision.ops import box_convert
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np

img_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
img_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

if not ("DISPLAY" in os.environ):
    import matplotlib as mpl

    mpl.use("Agg")

def annotate(image_source: np.ndarray, boxes: torch.Tensor, phrases: List[str]) -> np.ndarray:
    """
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates.
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    # 可视化不超限
    xyxy = np.clip(xyxy, a_min=1.0, a_max=(min(h, w) - 1.0))
    detections = sv.Detections(xyxy=xyxy)

    labels = []
    for p in phrases:
        labels.append(f'DBH: {p[1]:.2f} cm, TH: {p[0] / 100.0:.2f} m')
    box_annotator = sv.BoxAnnotator()
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    return annotated_frame # h, w, 3

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def save_obj(rgb, position, obj, path):
    if isinstance(obj, torch.Tensor):
        obj = obj.cpu().numpy()
    phrases_pred = obj
    rgb = rgb.detach()
    rgb.mul_(img_std.type_as(rgb)).add_(img_mean.type_as(rgb))
    rgb = rgb[0].data.cpu().numpy()
    r = rgb.transpose(1, 2, 0) * 255.0
    position = position.cpu()
    o_pred = annotate(r, position, phrases_pred)
    image = o_pred.astype(np.uint8)
    cv2.imwrite(path, image)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_res(masks, input_box, filename, image):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    for i, mask in enumerate(masks):
        show_mask(mask, plt.gca(), random_color=True)
    plt.axis('off')
    plt.savefig(filename,bbox_inches='tight',pad_inches=-0.1)
    plt.close()