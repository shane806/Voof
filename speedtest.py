#!/Users/shanehumphrey/miniconda3/envs/pytorchCV_x86/bin/python
"""
Created on Thu Jan 21 05:45:52 2021

@author: shanehumphrey
"""

import cv2 as cv
import numpy as np
import torch
# from tinygrad.tensor import Tensor
import torchvision.transforms as T
from tqdm import tqdm


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def generate_frames(video):
    # (height, width, number_of_channels) = (480, 640, 3)
    # video = cv.VideoCapture(mp4_path)
    ret, prev_frame = video.read()

    while ret:
        ret, curr_frame = video.read()
        if ret:
            yield ret, prev_frame, curr_frame
        prev_frame = curr_frame


def transform(image, bright_factor):
    """augment brightness, crop/resize"""

    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    hsv_image[:, :, 2] = hsv_image[:, :, 2] * bright_factor

    image_rgb = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)

    resized_image = cv.resize(image_rgb[100:440, :-90,:], (330, 99),
                              interpolation=cv.INTER_AREA)

    return resized_image


def generate_optical_flow_dataset(mp4_path, text_path):
    """generate dataset from mp4 and txt"""

    # (height, width, number_of_channels) = (480, 640, 3)
    video = cv.VideoCapture(mp4_path)
    ret, prev = video.read()
    t = 0
    show_hsv = False
    show_rgb = False

    while ret:

        ret, curr = video.read()

        if ret:

            # yield ret, prev_frame, curr_frame

    # for t, (ret, prev_frame, curr_frame) in enumerate(tqdm(generate_frames(video),
    #                                                        desc='Generating dense optical flow tensors')):

            bright_factor = 0.2 + np.random.uniform()
    
            prev_frame, curr_frame = transform(prev, bright_factor), \
                                     transform(curr, bright_factor)
    
            prev_gray, curr_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY), \
                                   cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)
    
    
            flow = cv.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                               None, 0.5, 1, 15, 2, 5, 1.3, 0)
    
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            
            hsv = np.zeros_like(prev_frame)
    
            hsv[:, :, 0] = ang * (180 / np.pi / 2)

            hsv[:, :, 1] = 255
    
            hsv[:, :, 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    
            rgb_flow = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
    
            ch = cv.waitKey(15)
    
            if (ch == ord('q')) or not ret:
                
                cv.destroyAllWindows()
    
                video.release()
    
                break
    
            # if ch == ord('1'):
            cv.imshow('frame', curr_frame)
            # cv.imshow('Optical flow', draw_flow(curr_gray, flow))
    
            if show_hsv:
                cv.imshow('hsv', hsv)
    
            if show_rgb:
    
                cv.imshow('rgb_flow', rgb_flow)
    
            if ch == ord('1'):
    
                show_rgb = not show_rgb
    
                print('RGB flow visualization is', ['off', 'on'][show_hsv])
    
            if ch == ord('2'):
                show_hsv = not show_hsv
    
                print('HSV flow visualization is', ['on', 'off'][show_hsv])
    
            rgb_flow_tensor = T.ToTensor()(rgb_flow).unsqueeze(0)
    
            # if not t:
            #     flow_stack = rgb_flow_tensor
    
            # else:
            
            #     flow_stack = torch.cat([flow_stack, rgb_flow_tensor])
    
            prev = curr
        t += 1

    # can't estimate speed of first frame
    # speed_vector = np.loadtxt(text_path)[1:]
    # flow_dataset = torch.utils.data.TensorDataset(flow_stack,
    #                                               torch.from_numpy(speed_vector).float())
    # return flow_dataset


def save_whole_set(mp4_path, text_path, save_path):
    """save whole dataset"""
    # flow_dataset = \
    generate_optical_flow_dataset(mp4_path, text_path)
    # torch.save(flow_dataset, save_path)


if __name__ == '__main__':
    save_whole_set('train.mp4', './data/train.txt', 'datasets/farneback_train.pt')
