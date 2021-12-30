import logging


from torchvision import transforms
from torch.utils.data import DataLoader

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import torch.nn.functional as nnf

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageColor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from bbox_utils import BoundingBox
import albumentations as A
from albumentations.pytorch.transforms import ToTensor, ToTensorV2

from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

logger = logging.getLogger(__name__)


def initialize_model(args):
    # load an instance segmentation model pre-trained pre-trained on COCO
    device = "cpu"
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=True, box_nms_thresh=0.95, rpn_nms_thresh=0.95,
    )
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    num_classes = 4
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    detection_model = args.detection_model
    model.load_state_dict(torch.load(detection_model, map_location=torch.device("cpu")))
    model.to(device)
    model.eval()

    init_model_dict = {
        "detection_model": model,
        "classification_confidence_threshold": args.classification_confidence_threshold,
        "iou_threshold": args.iou_threshold,
        "mask_thresh": args.mask_thresh,
    }
    return init_model_dict


def detect_objects(
    img, model, iou_threshold, classification_confidence_threshold, mask_thresh
):
    pred = None
    if img is not None:
        trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        img_tensor = trans(img)
        pred = None
        if img_tensor.shape[0] == 3:
            with torch.no_grad():
                pred = model([img_tensor.to("cpu")])

            pred = {k: v.cpu().detach() for k, v in pred[0].items()}

            keep = torchvision.ops.nms(
                pred["boxes"], pred["scores"], iou_threshold=iou_threshold
            )

            pred = {k: v[keep] for k, v in pred.items()}

            mask = pred["scores"] > classification_confidence_threshold
            pred = {k: v[mask] for k, v in pred.items()}
            # Mask pixel thresh
            pred["masks"][pred["masks"] > mask_thresh] = 1
            pred["masks"][pred["masks"] < mask_thresh] = 0

    return pred


def normalize_polygon(image_shape, box_polygon):
    height, width, _ = image_shape

    coords = []
    for point_idx, point in enumerate(box_polygon):
        if len(box_polygon) > point_idx + 1:
            line = {
                "Start": {"X": str(point[0] / width), "Y": str(point[1] / height)},
                "End": {
                    "X": str(box_polygon[point_idx + 1][0] / width),
                    "Y": str(box_polygon[point_idx + 1][1] / height),
                },
            }

        else:
            line = {
                "Start": {"X": str(point[0] / width), "Y": str(point[1] / height)},
                "End": {
                    "X": str(box_polygon[0][0] / width),
                    "Y": str(box_polygon[0][1] / height),
                },
            }
        coords.append(line)

    return coords


def bbox_to_polygon(bbox):
    polygon_bbox = np.array(
        [
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
        ]
    )
    return polygon_bbox


def convert_xyxy_xywh(box):
    xy1 = tuple([int(box[0]), int(box[1])])
    xy2 = tuple([int(box[2]), int(box[3])])
    bbox = BoundingBox.from_xyxy(xy1, xy2)
    # Get XYWH
    xy, w, h = bbox.to_xywh()
    return [list(xy)[0], list(xy)[1], w, h]


def predict_one_sample(
    image,
    detection_model,
    iou_threshold,
    classification_confidence_threshold,
    mask_thresh,
):
    """
    :param image: (PIL.Image) loaded image
    :param model: (torch.nn.Module) trained model with loaded state dict
    :param id_to_class: (dict) dictionary mapping indices to class names

    :return: (dict) with preds 
    """
    dic2map = {"1": "clamp standoff bad", "2": "metal snap in", "3": "plastic support"}
    height, width, c = image.shape
    image_shape = image.shape
    pred = detect_objects(
        image,
        detection_model,
        iou_threshold,
        classification_confidence_threshold,
        mask_thresh,
    )
    if pred:
        predictions = []
        output = {"predictions": []}
        # Crop and Predcit each one
        for i, box in enumerate(pred["boxes"]):
            dic = {
                "probability": 0.0,
                "tagId": 1,
                "tagName": "string",
                "boundingBox": {"left": 0.0, "top": 0.0, "width": 0.0, "height": 0.0},
                "polygon": {},
            }
            dic["tagName"] = dic2map[str(pred["labels"][i].numpy())]
            dic["probability"] = float(str(pred["scores"][i].numpy()))
            box_xywh = convert_xyxy_xywh(box)
            dic["boundingBox"]["left"] = str(box_xywh[0] / width)
            dic["boundingBox"]["top"] = str(box_xywh[1] / height)
            dic["boundingBox"]["width"] = str(box_xywh[2] / width)
            dic["boundingBox"]["height"] = str(box_xywh[3] / height)
            polygon_bbox = bbox_to_polygon(box)
            dic["polygon"] = normalize_polygon(image_shape, polygon_bbox)
            predictions.append(dic)
        output["predictions"] = predictions

        return output
