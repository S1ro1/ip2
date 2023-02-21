import numpy as np
import cv2 as cv
import torch
import json
import matplotlib.cm as cm

from models.matching import Matching
from models.utils import *
from collections import defaultdict
torch.set_grad_enabled(False)


class SuperpointsExtractor:
  def __init__(self, video_path):
    self.video_path = video_path
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    self.config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.003,
            'max_keypoints': 10,
        },
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.4,
        }
    }

    self.matching = Matching(self.config).eval().to(self.device)
    self.keys = ['keypoints', 'scores', 'descriptors']

    self.kp_count = 0
    self.frame_id = 0
    self.extracted_markers = {}
    self.prev_data = None

    self.prev_point_ids = []

  def extract_features(self, frame):
    img = frame2tensor(frame, self.device)
    if self.prev_data is None:

      self.prev_data = self.matching.superpoint({'image': img})
      self.prev_data = {k + '0': self.prev_data[k] for k in self.keys}
      self.prev_data['image0'] = img

      self.prev_point_ids = [None] * self.prev_data['keypoints0'][0].shape[0]
      return None

    pred = self.matching({**self.prev_data, 'image1': img})
    kpts0 = self.prev_data['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()

    matches = pred['matches0'][0].cpu().numpy()

    for i in range(matches.shape[0]):
      if self.prev_point_ids[i] is None and matches[i] > -1:
        self.prev_point_ids[i] = self.kp_count
        self.kp_count += 1

    keypoint_results = {self.prev_point_ids[i]: (kp[0], kp[1]) for i, kp in enumerate(kpts0) if self.prev_point_ids[i] is not None}

    results = {[self.frame_id], [keypoint_results]}

    self.prev_data = {k + '0': pred[k + '1'] for k in self.keys}
    self.prev_data['image0'] = img

    new_point_ids = [None] * kpts1.shape[0]

    for i, id in enumerate(self.prev_point_ids):
      if id is not None:
        new_point_ids[matches[i]] = id

    self.prev_point_ids = new_point_ids

    return results
