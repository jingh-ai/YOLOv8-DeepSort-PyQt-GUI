import cv2 as cv
import numpy as np
from collections import namedtuple
from src.models.base.yolov8_base import YoloPredictorBase
from src.utils.visualize import PALLETE
import math


Model = namedtuple("Model", "model confidence_threshold iou_threshold input_size class_names")


class DetectorBase(YoloPredictorBase):
    @staticmethod    
    def draw_results(image, model_results):
        img_cpy = image.copy()
        if model_results == []:
            return img_cpy
        height, width, _ = img_cpy.shape

        for obj in model_results:
            x, y, w, h, angle = obj["bbox"][::]
            id = int(obj["id"])
            class_name = obj["class"]
            color = PALLETE[id%PALLETE.shape[0]]
            text = '%d-%s'%(id,class_name)
            txt_color_light = (255, 255, 255)
            txt_color_dark = (0, 0, 0)
            font = cv.FONT_HERSHEY_SIMPLEX
            FONT_SCALE = 1e-3 
            THICKNESS_SCALE = 6e-4 
            font_scale = min(width, height) * FONT_SCALE
            if font_scale <= 0.4:
                font_scale = 0.41 
            elif font_scale > 2:
                font_scale = 2.0
            thickness = math.ceil(min(width, height) * THICKNESS_SCALE)
            text_size = cv.getTextSize(text, font, font_scale, thickness)[0]

            angle_rad = math.radians(angle)
            c = math.cos(angle_rad)
            s = math.sin(angle_rad)
            # Get the box vertices
            vertices = np.array([
                [x - w * c - h * s, y + w * s - h * c],
                [x + w * c - h * s, y - w * s - h * c],
                [x + w * c + h * s, y - w * s + h * c],
                [x - w * c + h * s, y + w * s + h * c]
            ], np.int32)
            vertices = vertices.reshape((-1, 1, 2))
            # Draw the box
            img_cpy = cv.polylines(img_cpy, [vertices], isClosed=True, color=color, thickness=thickness)
            # Display class label
            cv.putText(img_cpy, text, (x - w // 2, y + h // 2), font, font_scale, txt_color_dark, thickness=thickness+1)
            cv.putText(img_cpy, text, (x - w // 2, y + h // 2), font, font_scale, txt_color_light, thickness=thickness)
        return img_cpy
    







