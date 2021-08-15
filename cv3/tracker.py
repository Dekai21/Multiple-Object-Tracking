import collections
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import motmetrics as mm
mm.lap.default_solver = 'lap'

import market.metrics as metrics

import os.path as osp

class Tracker:
	"""The main tracking file, here is where magic happens."""

	def __init__(self, obj_detect):
		self.obj_detect = obj_detect

		self.tracks = []
		self.track_num = 0
		self.im_index = 0
		self.results = {}

		self.mot_accum = None

	def reset(self, hard=True):
		self.tracks = []

		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0

	def add(self, new_boxes, new_scores):
		"""Initializes new Track objects and saves them."""
		num_new = len(new_boxes)
		for i in range(num_new):
			self.tracks.append(Track(
				new_boxes[i],
				new_scores[i],
				self.track_num + i
			))
		self.track_num += num_new

	def get_pos(self):
		"""Get the positions of all active tracks."""
		if len(self.tracks) == 1:
			box = self.tracks[0].box
		elif len(self.tracks) > 1:
			box = torch.stack([t.box for t in self.tracks], 0)
		else:
			box = torch.zeros(0).cuda()
		return box

	def update_results(self):
		# results
		for t in self.tracks:
			if t.id not in self.results.keys():
				self.results[t.id] = {}
			self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])])

		self.im_index += 1


	def data_association(self, boxes, scores):
		self.tracks = []
		self.add(boxes, scores)

	def step(self, frame):
		"""This function should be called every timestep to perform tracking with a blob
		containing the image information.
		"""
		
		# object detection
		#boxes, scores = self.obj_detect.detect(frame['img'])
		boxes, scores = frame['det']['boxes'], frame['det']['scores']


		self.data_association(boxes, scores)
		self.update_results()


	def get_results(self):
		return self.results


class ReIDTracker(Tracker):
	def add(self, new_boxes, new_scores, new_features):
		"""Initializes new Track objects and saves them."""
		num_new = len(new_boxes)
		for i in range(num_new):
			self.tracks.append(Track(
				new_boxes[i],
				new_scores[i],
				self.track_num + i,
				new_features[i]
			))
		self.track_num += num_new

	def reset(self, hard=True):
		self.tracks = []
		#self.inactive_tracks = []

		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0

	def data_association(self, boxes, scores, features):
		raise NotImplementedError

	def step(self, frame, next_frame):
		"""This function should be called every timestep to perform tracking with a blob
		containing the image information.
		"""
		boxes = frame['det']['boxes']
		scores = frame['det']['scores']
		reid_feats= frame['det']['reid'].cpu()

		boxes_2 = next_frame['det']['boxes']
		scores_2 = next_frame['det']['scores']
		reid_feats_2= next_frame['det']['reid'].cpu()
		self.data_association(boxes, scores, reid_feats, boxes_2, scores_2, reid_feats_2)

		# results
		self.update_results()


	def compute_distance_matrix(self, track_features, pred_features, track_boxes, boxes, metric_fn, alpha=0.0):
		UNMATCHED_COST = 255.0

		# Build cost matrix.
		distance = mm.distances.iou_matrix(track_boxes.numpy(), boxes.numpy(), max_iou=0.5)

		appearance_distance = metrics.compute_distance_matrix(track_features, pred_features, metric_fn=metric_fn)
		appearance_distance = appearance_distance.numpy() * 0.5
		# return appearance_distance

		assert np.alltrue(appearance_distance >= -0.1)
		assert np.alltrue(appearance_distance <= 1.1)

		combined_costs = alpha * distance + (1-alpha) * appearance_distance

		# Set all unmatched costs to _UNMATCHED_COST.
		distance = np.where(np.isnan(distance), UNMATCHED_COST, combined_costs)

		distance = np.where(appearance_distance > 0.1, UNMATCHED_COST, distance)

		return distance


class Track(object):
	"""This class contains all necessary for every individual track."""

	def __init__(self, box, score, track_id, feature=None, inactive=0):
		self.id = track_id
		self.box = box
		self.score = score
		self.feature = collections.deque([feature])
		self.inactive = inactive
		self.max_features_num = 10


	def add_feature(self, feature):
		"""Adds new appearance features to the object."""
		self.feature.append(feature)
		if len(self.feature) > self.max_features_num:
			self.feature.popleft()

	def get_feature(self):
		if len(self.feature) > 1:
			feature = torch.stack(list(self.feature), dim=0)
		else:
			feature = self.feature[0].unsqueeze(0)
		#return feature.mean(0, keepdim=False)
		return feature[-1]

############