import cv2
import numpy as np
from os.path import join, exists
import os


def mask_plot_label(mask, label, color):
    c_mask = mask.astype(np.uint8)
    z_channel = np.zeros(256, np.dtype('uint8'))
    color_map = np.dstack((z_channel, z_channel, z_channel))
    color_map[:, label, :] = color
    v_mask = np.dstack((c_mask, c_mask, c_mask))
    color_mask = cv2.LUT(v_mask, color_map)
    return color_mask


def blend_img_mask(image, color_mask_3ch, alpha=0.5):
    blend_v_img = cv2.addWeighted(color_mask_3ch, alpha, image, (1 - alpha), 0, dtype=cv2.CV_8UC3)

    img_mask = ((color_mask_3ch[:, :, 0] == 0) & (color_mask_3ch[:, :, 1] == 0) & (color_mask_3ch[:, :, 2] == 0))
    img_mask = img_mask.astype(np.int32)
    img_mask_inv = abs(img_mask - 1)
    img_mask = img_mask.astype(np.uint8)
    img_mask_inv = img_mask_inv.astype(np.uint8)

    img_vis_fg = np.zeros_like(blend_v_img)
    img_vis_fg[:, :, 0] = blend_v_img[:, :, 0] * img_mask_inv
    img_vis_fg[:, :, 1] = blend_v_img[:, :, 1] * img_mask_inv
    img_vis_fg[:, :, 2] = blend_v_img[:, :, 2] * img_mask_inv

    img_vis_bg = np.zeros_like(blend_v_img)
    img_vis_bg[:, :, 0] = image[:, :, 0] * img_mask
    img_vis_bg[:, :, 1] = image[:, :, 1] * img_mask
    img_vis_bg[:, :, 2] = image[:, :, 2] * img_mask

    img_vis = cv2.add(img_vis_fg, img_vis_bg)
    return img_vis


def draw_mask(data, data_mask, colors: list, alpha=0.3):
    if isinstance(data, str) and isinstance(data_mask, str):
        assert exists(data)
        if not exists(data_mask):
            print("not mask file={}".format(data_mask))
            return cv2.imread(data)
        image = cv2.imread(data)
        mask = cv2.imread(data_mask, 0)
    else:
        image = data
        mask = data_mask
    labels = list(np.unique(mask))
    labels.remove(0)
    for label in labels:
        temp = np.zeros_like(mask)
        temp[mask == label] = label
        color_mask = mask_plot_label(temp, label, colors[label])
        image = blend_img_mask(image, color_mask, alpha=alpha)
    return image