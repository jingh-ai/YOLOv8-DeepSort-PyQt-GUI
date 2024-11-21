import numpy as np
import cv2 as cv
import math


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr, rotated=False):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    
    if rotated:
        keep = nms_rotated(valid_boxes, valid_scores, nms_thr)
    else:
        keep = nms(valid_boxes, valid_scores, nms_thr)
    keep = keep[:10]
    dets = []
    dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1)
    print("box")
    print(valid_boxes[keep])
    print("score")
    print(valid_scores[keep, None])
    print("class")
    print(valid_cls_inds[keep, None])
    return dets

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    return keep

def get_covariance_matrix_np(boxes):
    gbbs = np.concatenate((boxes[:, 2:4]**2 / 12, boxes[:, 4:]), axis=-1)
    a, b, c = np.split(gbbs, [1, 2], axis=-1)
    cos = np.cos(c)
    sin = np.sin(c)
    cos2 = cos**2
    sin2 = sin**2
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

def batch_probiou(obb1, obb2, eps=1e-7):
    x1, y1 = np.split(obb1[..., :2], 2, axis=-1)
    x2 = obb2[..., 0].reshape(1, -1)
    y2 = obb2[..., 1].reshape(1, -1)
    a1, b1, c1 = get_covariance_matrix_np(obb1)
    a2, b2, c2 = [get_covariance_matrix_np(obb2)[i].reshape(1, -1) for i in range(3)]

    t1 = (
        ((a1 + a2) * (y1 - y2)**2 + (b1 + b2) * (x1 - x2)**2) / ((a1 + a2) * (b1 + b2) - (c1 + c2)**2 + eps)
    ) * 0.25

    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2)**2 + eps)) * 0.5
    t3 = np.log((
        ((a1 + a2) * (b1 + b2) - (c1 + c2)**2)
        / (4 * (np.sqrt((np.maximum(0, a1 * b1 - c1**2)) * (np.maximum(0, a2 * b2 - c2**2))) + eps)
           + eps)
    )) * 0.5
    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1 - hd


def nms_rotated(boxes, scores, threshold=0.45):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    sorted_idx = np.argsort(scores)[::-1]
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes)
    np.fill_diagonal(ious, 0)
    pick = np.where(np.max(ious, axis=0) <= threshold)[0]
    return sorted_idx[pick]


def multiclass_nms_class_agnostic_keypoints(boxes, scores, kpts, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_boxes = xywh2xyxy(valid_boxes)
    valid_cls_inds = cls_inds[valid_score_mask]
    valid_kpts = kpts[valid_score_mask]

    keep = nms(valid_boxes, valid_scores, nms_thr)
    dets = []
    for i in keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None] ,valid_kpts[keep]], 1
        )
    return dets

