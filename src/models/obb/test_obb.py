import numpy as np
import torch

# Numpy version
def _get_covariance_matrix_np(boxes):
    gbbs = np.concatenate((boxes[:, 2:4]**2 / 12, boxes[:, 4:]), axis=-1)
    a, b, c = np.split(gbbs, [1, 2], axis=-1)
    cos = np.cos(c)
    sin = np.sin(c)
    cos2 = cos**2
    sin2 = sin**2
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

def probiou_np(obb1, obb2, eps=1e-7):
    x1, y1 = np.split(obb1[..., :2], 2, axis=-1)    #obb1[..., :2].split(1, dim=-1)
    x2, y2 = np.split(obb2[..., :2], 2, axis=-1)
    a1, b1, c1 = _get_covariance_matrix_np(obb1)
    a2, b2, c2 = _get_covariance_matrix_np(obb2)

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
    iou = 1 - hd
    return iou

def batch_probiou_np(obb1, obb2, eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
    """
    x1, y1 = np.split(obb1[..., :2], 2, axis=-1)    #obb1[..., :2].split(1, dim=-1)
    print("x1 ", x1.shape)
    print("y1 ", y1.shape)
    #x2, y2 = np.split(obb2[..., :2], 2, axis=-1)
    x2 = obb2[..., 0].reshape(1, -1)
    y2 = obb2[..., 1].reshape(1, -1)
    print("x2 ", x2.shape)
    print("y2 ", y2.shape)
    a1, b1, c1 = _get_covariance_matrix_np(obb1)
    print("a1 ", a1.shape)
    print("b1 ", b1.shape)
    print("c1 ", c1.shape)
    #a2, b2, c2 = _get_covariance_matrix_np(obb2)
    a2, b2, c2 = [_get_covariance_matrix_np(obb2)[i].reshape(1, -1) for i in range(3)]
    print("a2 ", a2.shape)
    print("b2 ", b2.shape)
    print("c2 ", c2.shape)


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

# PyTorch version
def _get_covariance_matrix(boxes):
    """
    Generating covariance matrix from obbs.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance metrixs corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

def probiou(obb1, obb2, CIoU=False, eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor): A tensor of shape (N, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, ) representing obb similarities.
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    if CIoU:  # only include the wh aspect ratio part
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU
    return iou

def batch_probiou(obb1, obb2, eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    print("x1 ", x1.size())
    print("y1 ", y1.size())
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    print("x2 ", x2.size())
    print("y2 ", y2.size())
    a1, b1, c1 = _get_covariance_matrix(obb1)
    print("a1 ", a1.size())
    print("b1 ", b1.size())
    print("c1 ", c1.size())
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))
    print("a2 ", a2.size())
    print("b2 ", b2.size())
    print("c2 ", c2.size())

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd

def nms_rotated(boxes, scores, threshold=0.45):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    sorted_idx = np.argsort(scores)[::-1]
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes)
    ious = np.fill_diagonal(ious, 0)  # Set self-intersection IoUs to 0
    pick = np.where(np.max(ious, axis=0) < threshold)[0]
    return sorted_idx[pick]

# Example usage
boxes_np = np.array([[100, 100, 50, 30, np.pi/2], [120, 120, 50, 30, np.pi/4], [140, 140, 50, 30, np.pi/3]])
boxes_torch = torch.tensor(boxes_np)

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("get_covariance_matrix")
print("Pytorch")
iou_torch = _get_covariance_matrix(boxes_torch)

print("Numpy")
iou_np = _get_covariance_matrix_np(boxes_np)

print(np.array_equal(iou_np[0], iou_torch[0].detach().cpu().numpy()))
print(np.array_equal(iou_np[1], iou_torch[1].detach().cpu().numpy()))
print(np.array_equal(iou_np[2], iou_torch[2].detach().cpu().numpy()))

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("batch_probiou")
print("Pytorch")
iou_torch = batch_probiou(boxes_torch, boxes_torch)
print("Numpy")
iou_np = batch_probiou_np(boxes_np, boxes_np)

iou_torch = iou_torch.detach().cpu().numpy()
print(np.array_equal(iou_np, iou_torch))
print("Numpy")
print(iou_np)
print("Pytorch")
print(iou_torch)
