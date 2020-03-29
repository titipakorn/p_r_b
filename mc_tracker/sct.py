"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import random
from copy import deepcopy as copy
from collections import namedtuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine, cdist

from PIL import Image

import torch

from utils.misc import none_to_zero
from shapely.geometry import Polygon, Point

from torchvision import transforms

THE_BIGGEST_DISTANCE = 10.

TrackedObj = namedtuple('TrackedObj', 'rect label')


class ClusterFeature:
    def __init__(self, feature_len, initial_feature=None):
        self.clusters = []
        self.clusters_sizes = []
        self.feature_len = feature_len
        if initial_feature is not None:
            self.clusters.append(initial_feature)
            self.clusters_sizes.append(1)

    def update(self, feature_vec):
        if len(self.clusters) < self.feature_len:  # not full cluster yet
            self.clusters.append(feature_vec)
            self.clusters_sizes.append(1)
        elif sum(self.clusters_sizes) < 2*self.feature_len:  # amount of features less than 2*size
            idx = random.randint(0, self.feature_len - 1)
            self.clusters_sizes[idx] += 1
            self.clusters[idx] += (feature_vec - self.clusters[idx]) / \
                self.clusters_sizes[idx]
            # calcualte average feature of random cluster
        else:
            distances = cdist(feature_vec.reshape(1, -1),
                              np.array(self.clusters).reshape(len(self.clusters), -1), 'cosine')
            nearest_idx = np.argmin(distances)
            self.clusters_sizes[nearest_idx] += 1
            self.clusters[nearest_idx] += (feature_vec - self.clusters[nearest_idx]) / \
                self.clusters_sizes[nearest_idx]

    def get_clusters_matrix(self):
        return np.array(self.clusters).reshape(len(self.clusters), -1)

    def __len__(self):
        return len(self.clusters)


def clusters_distance(clusters1, clusters2):
    if len(clusters1) > 0 and len(clusters2) > 0:
        distances = 0.5 * cdist(clusters1.get_clusters_matrix(),
                                clusters2.get_clusters_matrix(), 'cosine')
        return np.amin(distances)
    return 1


def clusters_vec_distance(clusters, feature):
    try:
        if len(clusters) > 0 and feature is not None and feature.shape[0] == 2048:
            distances = 0.5 * cdist(clusters.get_clusters_matrix(),
                                    feature.reshape(1, -1), 'cosine')
            return np.amin(distances)
    except:
        pass
    return 1.


def convert_polygon(json):
    y = []
    for i in json:
        # 320,544
        # y.append(tuple((i["x"]*(320/1280), i["y"]*(544/720))))
        y.append(tuple((i["x"], i["y"])))
    return y


class SingleCameraTracker:
    COUNT_IN = 0
    COUNT_OUT = 0

    def __init__(self, id, global_id_getter, global_id_releaser,
                 reid_model=None,
                 time_window=10,
                 continue_time_thresh=2,
                 track_clear_thresh=3000,
                 match_threshold=0.4,
                 merge_thresh=0.35,
                 n_clusters=4,
                 rectify_thresh=0.1,
                 max_bbox_velocity=0.2,
                 detection_occlusion_thresh=0.7,
                 track_detection_iou_thresh=0.5, p_in=[], p_out=[]):
        self.out_poly = Polygon(tuple(convert_polygon(p_in)))
        self.in_poly = Polygon(tuple(convert_polygon(p_out)))
        self.reid_model = reid_model
        self.global_id_getter = global_id_getter
        self.global_id_releaser = global_id_releaser
        self.id = id
        self.tracks = []
        self.history_tracks = []
        self.time = 0
        self.candidates = []
        self.time_window = time_window
        self.continue_time_thresh = continue_time_thresh
        self.track_clear_thresh = track_clear_thresh
        self.match_threshold = match_threshold
        self.merge_thresh = merge_thresh
        self.rectify_thresh = rectify_thresh
        self.n_clusters = n_clusters
        self.max_bbox_velocity = max_bbox_velocity
        self.detection_occlusion_thresh = detection_occlusion_thresh
        self.track_detection_iou_thresh = track_detection_iou_thresh
        self.data_transform = transforms.Compose([
            transforms.Resize([64, 128]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),

        ])

    def process(self, images, detections, mask=None):
        reid_features = [None]*len(images)
        if self.reid_model:
            reid_features = self._get_embeddings(images, mask)

        assignment = self._continue_tracks(images, detections, reid_features)
        if self.time % 2 == 0:
            self._create_new_tracks(
                images, detections, reid_features, assignment)
            self._merge_tracks()
        self.time += 1

    def get_tracked_objects(self):
        objs = []
        label = 'ID'
        objs = []
        for track in self.tracks:
            if track['timestamps'][-1] == self.time - 1 and len(track['timestamps']) > self.time_window:
                objs.append(TrackedObj(track['boxes'][-1],
                                       label + ' ' + str(track['id'])))
        return objs

    def get_tracks(self):
        return self.tracks

    def get_archived_tracks(self):
        return self.history_tracks

    def check_and_merge(self, track_source, track_candidate):
        id_candidate = track_source['id']
        idx = -1
        for i, track in enumerate(self.tracks):
            if track['boxes'] == track_candidate['boxes']:
                idx = i
        if idx < 0:  # in this case track already has been modified, merge is invalid
            return

        collisions_found = False
        for i, track in enumerate(self.tracks):
            if track is not None and track['id'] == id_candidate:
                if self.tracks[i]['timestamps'][-1] <= self.tracks[idx]['timestamps'][0] or \
                        self.tracks[idx]['timestamps'][-1] <= self.tracks[i]['timestamps'][0]:
                    self.tracks[i]['id'] = id_candidate
                    self.tracks[idx]['id'] = id_candidate
                    self._concatenate_tracks(i, idx)
                collisions_found = True

        if not collisions_found:
            self.tracks[idx]['id'] = id_candidate
            new_clusters = self._merge_clustered_features(self.tracks[idx]['f_cluster'],
                                                          track_source['f_cluster'],
                                                          self.tracks[idx]['features'],
                                                          track_source['features'])
            self.tracks[idx]['f_cluster'] = copy(new_clusters)
            track_candidate['f_cluster'] = copy(new_clusters)
        self.tracks = list(filter(None, self.tracks))

    def tlbr_to_tlwh(self, boxes):
        ret = np.array(boxes)
        np.subtract(ret[2:], ret[:2], out=ret[2:], casting="unsafe")
        return ret

    def tlwh_to_xyah(self, boxes):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.array(boxes)
        np.add(ret[:2], ret[2:] / 2, out=ret[:2], casting="unsafe")
        ret[2] /= ret[3]
        return ret

    def _continue_tracks(self, frames, detections, features):
        active_tracks_idx = []
        for i, track in enumerate(self.tracks):
            if track['timestamps'][-1] >= self.time - self.continue_time_thresh:
                active_tracks_idx.append(i)

        occluded_det_idx = []
        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections):
                if i != j and self._ios(det1, det2) > self.detection_occlusion_thresh:
                    occluded_det_idx.append(i)
                    features[i] = None
                    break

        cost_matrix = self._compute_detections_assignment_cost(
            active_tracks_idx, detections, features)

        assignment = [None for _ in range(cost_matrix.shape[0])]
        if cost_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for i, j in zip(row_ind, col_ind):
                idx = active_tracks_idx[j]
                if cost_matrix[i, j] < self.match_threshold and \
                    self._check_velocity_constraint(self.tracks[idx], detections[i]) and \
                        self._giou(self.tracks[idx]['boxes'][-1], detections[i]) > self.track_detection_iou_thresh:
                    assignment[i] = j

            for i, j in enumerate(assignment):  # candidates
                if j is not None:
                    idx = active_tracks_idx[j]
                    boxes = self.tlwh_to_xyah(
                        self.tlbr_to_tlwh(detections[i]))
                    current_point = Point((boxes[0], boxes[1]))  # center

                    if(current_point.within(self.in_poly)):
                        if(self.tracks[idx]['in_status'] == False):
                            self.tracks[idx]['in_status'] = True
                            c_in_temp = SingleCameraTracker.COUNT_IN
                            if(len(self.candidates) > 0):
                                list_curr_f = [feature['features'][-1]
                                               for feature in self.candidates]
                                list_avg_f = [feature['avg_feature']
                                              for feature in self.candidates]
                                list_clus_f = [clusters_vec_distance(feature['f_cluster'], self.tracks[idx]['features'][-1])
                                               for feature in self.candidates]
                                distances_f = cdist(self.tracks[idx]['features'][-1].reshape(1, -1),
                                                    np.array(list_curr_f).reshape(len(list_curr_f), -1), 'cosine')
                                nearest_idx_f = np.argmin(distances_f)
                                distances_avg_f = cdist(self.tracks[idx]['features'][-1].reshape(1, -1),
                                                        np.array(list_avg_f).reshape(len(list_avg_f), -1), 'cosine')
                                nearest_idx_avg = np.argmin(distances_avg_f)
                                nearest_idx_clus = list_clus_f.index(
                                    min(list_clus_f))
                                distances = [
                                    distances_f[nearest_idx_f], distances_avg_f[nearest_idx_avg], list_clus_f[nearest_idx_clus]]
                                f_indexes = [nearest_idx_f,
                                             nearest_idx_avg, nearest_idx_clus]
                                f_dist = distances.index(min(distances))
                                if(distances[f_dist] < self.rectify_thresh):
                                    self.candidates[f_indexes[f_dist]]['boxes'].append(
                                        self.tracks[idx]['boxes'][-1])
                                    self.candidates[f_indexes[f_dist]]['timestamps'].append(
                                        self.tracks[idx]['timestamps'][-1])
                                    self.candidates[f_indexes[f_dist]]['features'].append(
                                        self.tracks[idx]['features'][-1])
                                    if self.tracks[idx]['features'][-1] is not None:
                                        self.candidates[f_indexes[f_dist]]['f_cluster'].update(
                                            features[i])
                                        if self.candidates[f_indexes[f_dist]]['avg_feature'] is None:
                                            self.candidates[f_indexes[f_dist]]['avg_feature'] = np.zeros(
                                                self.tracks[idx]['features'][-1].shape)
                                        self.candidates[f_indexes[f_dist]]['avg_feature'] += (self.tracks[idx]['features'][-1] - self.candidates[f_indexes[f_dist]]['avg_feature']) / \
                                            len(self.tracks[idx]
                                                ['features'])
                                    else:
                                        self.candidates[f_indexes[f_dist]
                                                        ]['avg_feature'] = None
                                    if(self.candidates[f_indexes[f_dist]]['out_status'] and self.candidates[f_indexes[f_dist]]['in_count'] is None):
                                        self.candidates[f_indexes[f_dist]
                                                        ]['in_count'] = 1
                                        img = Image.open(frames[i].file)
                                        img.save(
                                            "extract_person/IN_{}.jpg".format(self.time+self.candidates[f_indexes[f_dist]]['id']))
                                        SingleCameraTracker.COUNT_IN += 1
                                    else:
                                        self.candidates[f_indexes[f_dist]
                                                        ]['in_status'] = True
                                        c_in_temp = -1
                                else:
                                    for track2 in self.candidates:
                                        if (self.tracks[idx]['timestamps'][0] > track2['timestamps'][-1]
                                                or track2['timestamps'][0] > self.tracks[idx]['timestamps'][-1]):
                                            if(self._giou(self.tracks[-1]['boxes'][-1], track2['boxes'][-1]) > self.track_detection_iou_thresh):
                                                track2['boxes'].append(
                                                    self.tracks[-1]['boxes'][-1])
                                                track2['timestamps'].append(
                                                    self.tracks[-1]['timestamps'][-1])
                                                track2['features'].append(
                                                    self.tracks[-1]['features'][-1])
                                                if self.tracks[-1]['features'][-1] is not None:
                                                    track2['f_cluster'].update(
                                                        features[i])
                                                    if track2['avg_feature'] is None:
                                                        track2['avg_feature'] = np.zeros(
                                                            self.tracks[-1]['features'][-1].shape)
                                                    track2['avg_feature'] += (self.tracks[-1]['features'][-1] - track2['avg_feature']) / \
                                                        len(self.tracks[-1]
                                                            ['features'])
                                                else:
                                                    track2['avg_feature'] = None
                                                track2['in_status'] = True
                                                c_in_temp = -1
                                                break
                            if(c_in_temp == SingleCameraTracker.COUNT_IN):
                                self.candidates.append(
                                    copy(self.tracks[idx]))
                    if(current_point.within(self.out_poly)):
                        if(self.tracks[idx]['out_status'] == False):
                            self.tracks[idx]['out_status'] = True
                            c_out_temp = SingleCameraTracker.COUNT_OUT
                            if(len(self.candidates) > 0):
                                list_curr_f = [feature['features'][-1]
                                               for feature in self.candidates]
                                list_avg_f = [feature['avg_feature']
                                              for feature in self.candidates]
                                list_clus_f = [clusters_vec_distance(feature['f_cluster'], self.tracks[idx]['features'][-1])
                                               for feature in self.candidates]
                                distances_f = cdist(self.tracks[idx]['features'][-1].reshape(1, -1),
                                                    np.array(list_curr_f).reshape(len(list_curr_f), -1), 'cosine')
                                nearest_idx_f = np.argmin(distances_f)
                                distances_avg_f = cdist(self.tracks[idx]['features'][-1].reshape(1, -1),
                                                        np.array(list_avg_f).reshape(len(list_avg_f), -1), 'cosine')
                                nearest_idx_avg = np.argmin(distances_avg_f)
                                nearest_idx_clus = list_clus_f.index(
                                    min(list_clus_f))
                                distances = [
                                    distances_f[nearest_idx_f], distances_avg_f[nearest_idx_avg], list_clus_f[nearest_idx_clus]]
                                f_indexes = [nearest_idx_f,
                                             nearest_idx_avg, nearest_idx_clus]
                                f_dist = distances.index(min(distances))
                                if(distances[f_dist] < self.rectify_thresh):
                                    self.candidates[f_indexes[f_dist]]['boxes'].append(
                                        self.tracks[idx]['boxes'][-1])
                                    self.candidates[f_indexes[f_dist]]['timestamps'].append(
                                        self.tracks[idx]['timestamps'][-1])
                                    self.candidates[f_indexes[f_dist]]['features'].append(
                                        self.tracks[idx]['features'][-1])
                                    if self.tracks[idx]['features'][-1] is not None:
                                        self.candidates[f_indexes[f_dist]]['f_cluster'].update(
                                            features[i])
                                        if self.candidates[f_indexes[f_dist]]['avg_feature'] is None:
                                            self.candidates[f_indexes[f_dist]]['avg_feature'] = np.zeros(
                                                self.tracks[idx]['features'][-1].shape)
                                        self.candidates[f_indexes[f_dist]]['avg_feature'] += (self.tracks[idx]['features'][-1] - self.candidates[f_indexes[f_dist]]['avg_feature']) / \
                                            len(self.tracks[idx]
                                                ['features'])
                                    else:
                                        self.candidates[f_indexes[f_dist]
                                                        ]['avg_feature'] = None
                                    if(self.candidates[f_indexes[f_dist]]['out_status'] and self.candidates[f_indexes[f_dist]]['in_count'] is None):
                                        self.candidates[f_indexes[f_dist]
                                                        ]['in_count'] = 1
                                        img = Image.open(frames[i].file)
                                        img.save(
                                            "extract_person/IN_{}.jpg".format(self.time+self.candidates[f_indexes[f_dist]]['id']))
                                        SingleCameraTracker.COUNT_IN += 1
                                    else:
                                        self.candidates[f_indexes[f_dist]
                                                        ]['in_status'] = True
                                        c_out_temp = -1
                                else:
                                    for track2 in self.candidates:
                                        if (self.tracks[idx]['timestamps'][0] > track2['timestamps'][-1]
                                                or track2['timestamps'][0] > self.tracks[idx]['timestamps'][-1]):
                                            if(self._giou(self.tracks[-1]['boxes'][-1], track2['boxes'][-1]) > self.track_detection_iou_thresh):
                                                track2['boxes'].append(
                                                    self.tracks[-1]['boxes'][-1])
                                                track2['timestamps'].append(
                                                    self.tracks[-1]['timestamps'][-1])
                                                track2['features'].append(
                                                    self.tracks[-1]['features'][-1])
                                                if self.tracks[-1]['features'][-1] is not None:
                                                    track2['f_cluster'].update(
                                                        features[i])
                                                    if track2['avg_feature'] is None:
                                                        track2['avg_feature'] = np.zeros(
                                                            self.tracks[-1]['features'][-1].shape)
                                                    track2['avg_feature'] += (self.tracks[-1]['features'][-1] - track2['avg_feature']) / \
                                                        len(self.tracks[-1]
                                                            ['features'])
                                                else:
                                                    track2['avg_feature'] = None
                                                track2['in_status'] = True
                                                c_out_temp = -1
                                                break
                            if(c_out_temp == SingleCameraTracker.COUNT_OUT):
                                self.candidates.append(
                                    copy(self.tracks[idx]))

                    self.tracks[idx]['boxes'].append(detections[i])
                    self.tracks[idx]['timestamps'].append(self.time)
                    try:
                        if features[i] is not None and features[i].shape[0] == 2048:
                            self.tracks[idx]['features'].append(features[i])
                            self.tracks[idx]['f_cluster'].update(features[i])
                            if self.tracks[idx]['avg_feature'] is None:
                                self.tracks[idx]['avg_feature'] = np.zeros(
                                    features[i].shape)
                            self.tracks[idx]['avg_feature'] += (features[i] - self.tracks[idx]['avg_feature']) / \
                                len(self.tracks[idx]['features'])
                        else:
                            self.tracks[idx]['avg_feature'] = None
                    except:
                        pass

        return assignment

    def _get_rectification_distance(self, track1, track2):
        if (track1['timestamps'][0] > track2['timestamps'][-1]
            or track2['timestamps'][0] > track1['timestamps'][-1]) \
                and track1['avg_feature'] is not None and track2['avg_feature'] is not None \
                and self._check_velocity_constraint(track1, track2['boxes'][-1]):
            return clusters_distance(
                track1['f_cluster'], track2['f_cluster'])
        return THE_BIGGEST_DISTANCE

    def _merge_tracks(self):
        clear_tracks = []
        for track in self.tracks:
            # remove too old tracks
            if track['timestamps'][-1] < self.time - self.track_clear_thresh:
                track['features'] = []
                self.history_tracks.append(track)
                continue
            # remove too short and outdated tracks
            if track['timestamps'][-1] < self.time - self.continue_time_thresh \
                    and len(track['timestamps']) < self.time_window:
                self.global_id_releaser(track['id'])
                continue
            clear_tracks.append(track)
        self.tracks = clear_tracks

        clear_candidates = []
        for person in self.candidates:
            if person['timestamps'][-1] < self.time - self.track_clear_thresh:
                continue
            if person['timestamps'][-1] < self.time - self.continue_time_thresh \
                    and len(person['timestamps']) < self.time_window:
                continue
            clear_candidates.append(person)
        self.candidates = clear_candidates
        # Clear Candidates
        active_tracks_idx = []
        not_active_tracks_idx = []
        for i, track in enumerate(self.candidates):
            if track['timestamps'][-1] >= self.time - (self.continue_time_thresh * 4) \
                    and len(track) >= (self.time_window // 2):
                active_tracks_idx.append(i)
            elif len(track) >= self.time_window // 2:
                not_active_tracks_idx.append(i)

        distance_matrix = np.zeros((len(active_tracks_idx),
                                    len(not_active_tracks_idx)), dtype=np.float32)
        for i, idx1 in enumerate(active_tracks_idx):
            for j, idx2 in enumerate(not_active_tracks_idx):
                distance_matrix[i, j] = self._get_rectification_distance(
                    self.tracks[idx1], self.tracks[idx2])

        indices_rows = np.arange(distance_matrix.shape[0])
        indices_cols = np.arange(distance_matrix.shape[1])

        while len(indices_rows) > 0 and len(indices_cols) > 0:
            i, j = np.unravel_index(
                np.argmin(distance_matrix), distance_matrix.shape)
            dist = distance_matrix[i, j]
            if dist < self.rectify_thresh:
                self._concatenate_tracks(active_tracks_idx[indices_rows[i]],
                                         not_active_tracks_idx[indices_cols[j]])
                distance_matrix = np.delete(distance_matrix, i, 0)
                indices_rows = np.delete(indices_rows, i)
                distance_matrix = np.delete(distance_matrix, j, 1)
                indices_cols = np.delete(indices_cols, j)
            else:
                break
        self.candidates = list(filter(None, self.candidates))

        # Clear Track
        distance_matrix = THE_BIGGEST_DISTANCE * \
            np.eye(len(self.tracks), dtype=np.float32)
        for i, track1 in enumerate(self.tracks):
            for j, track2 in enumerate(self.tracks):
                if j >= i:
                    break
                distance_matrix[i, j] = self._get_rectification_distance(
                    track1, track2)

        distance_matrix += np.transpose(distance_matrix)

        assignment = [None]*distance_matrix.shape[0]
        indices_rows = np.arange(distance_matrix.shape[0])
        indices_cols = np.arange(distance_matrix.shape[1])

        while len(indices_rows) > 0 and len(indices_cols) > 0:
            i, j = np.unravel_index(
                np.argmin(distance_matrix), distance_matrix.shape)
            dist = distance_matrix[i, j]
            if dist < self.merge_thresh:
                assignment[indices_rows[i]] = indices_cols[j]
                distance_matrix = np.delete(distance_matrix, i, 0)
                distance_matrix = np.delete(distance_matrix, j, 1)
                indices_rows = np.delete(indices_rows, i)
                indices_cols = np.delete(indices_cols, j)
            else:
                break

        for i, idx in enumerate(assignment):
            if idx is not None and self.tracks[idx] is not None and self.tracks[i] is not None:
                self._concatenate_tracks(self.tracks[i], self.tracks[idx])

        self.tracks = list(filter(None, self.tracks))

    def _concatenate_tracks(self, track1, track2):
        if track1['timestamps'][-1] <= track2['timestamps'][0]:
            track1['avg_feature'] = (none_to_zero(track1['avg_feature'])*len(track1['features']) +
                                     none_to_zero(track2['avg_feature'])*len(track2['features']))
            track1['f_cluster'] = self._merge_clustered_features(
                track1['f_cluster'],
                track2['f_cluster'],
                track1['avg_feature'],
                track2['avg_feature'])
            track1['timestamps'] += track2['timestamps']
            track1['boxes'] += track2['boxes']
            track1['features'] += track2['features']
            track1['avg_feature'] /= len(track1['features'])
            track2 = None
        else:
            assert track2['timestamps'][-1] <= track1['timestamps'][0]
            track2['avg_feature'] = (none_to_zero(track1['avg_feature'])*len(track1['features']) +
                                     none_to_zero(track2['avg_feature'])*len(track2['features']))
            track2['f_cluster'] = self._merge_clustered_features(
                track1['f_cluster'],
                track2['f_cluster'],
                track1['avg_feature'],
                track2['avg_feature'])
            track2['timestamps'] += track1['timestamps']
            track2['boxes'] += track1['boxes']
            track2['features'] += track1['features']
            track2['avg_feature'] /= len(
                track2['features'])
            track1 = None

    def _create_new_tracks(self, frames, detections, features, assignment):
        assert len(detections) == len(features)
        for i, j in enumerate(assignment):
            if j is None:
                self.tracks.append(self._create_tracklet_descr(self.time,
                                                               detections[i],
                                                               self.global_id_getter(),
                                                               features[i]))
                boxes = self.tlwh_to_xyah(
                    self.tlbr_to_tlwh(detections[i]))
                current_point = Point((boxes[0], boxes[1]))  # center

                if(current_point.within(self.in_poly)):
                    if(self.tracks[-1]['in_status'] == False):
                        self.tracks[-1]['in_status'] = True
                        c_in_temp = SingleCameraTracker.COUNT_IN
                        if(len(self.candidates) > 0):
                            list_curr_f = [feature['features'][-1]
                                           for feature in self.candidates]
                            list_avg_f = [feature['avg_feature']
                                          for feature in self.candidates]
                            list_clus_f = [clusters_vec_distance(feature['f_cluster'], self.tracks[-1]['features'][-1])
                                           for feature in self.candidates]
                            distances_f = cdist(self.tracks[-1]['features'][-1].reshape(1, -1),
                                                np.array(list_curr_f).reshape(len(list_curr_f), -1), 'cosine')
                            nearest_idx_f = np.argmin(distances_f)
                            distances_avg_f = cdist(self.tracks[-1]['features'][-1].reshape(1, -1),
                                                    np.array(list_avg_f).reshape(len(list_avg_f), -1), 'cosine')
                            nearest_idx_avg = np.argmin(distances_avg_f)
                            nearest_idx_clus = list_clus_f.index(
                                min(list_clus_f))
                            distances = [
                                distances_f[nearest_idx_f], distances_avg_f[nearest_idx_avg], list_clus_f[nearest_idx_clus]]
                            f_indexes = [nearest_idx_f,
                                         nearest_idx_avg, nearest_idx_clus]
                            f_dist = distances.index(min(distances))
                            if(distances[f_dist] < self.rectify_thresh):
                                self.candidates[f_indexes[f_dist]]['boxes'].append(
                                    self.tracks[-1]['boxes'][-1])
                                self.candidates[f_indexes[f_dist]]['timestamps'].append(
                                    self.tracks[-1]['timestamps'][-1])
                                self.candidates[f_indexes[f_dist]]['features'].append(
                                    self.tracks[-1]['features'][-1])
                                if self.tracks[-1]['features'][-1] is not None:
                                    self.candidates[f_indexes[f_dist]]['f_cluster'].update(
                                        features[i])
                                    if self.candidates[f_indexes[f_dist]]['avg_feature'] is None:
                                        self.candidates[f_indexes[f_dist]]['avg_feature'] = np.zeros(
                                            self.tracks[-1]['features'][-1].shape)
                                    self.candidates[f_indexes[f_dist]]['avg_feature'] += (self.tracks[-1]['features'][-1] - self.candidates[f_indexes[f_dist]]['avg_feature']) / \
                                        len(self.tracks[-1]
                                            ['features'])
                                else:
                                    self.candidates[f_indexes[f_dist]
                                                    ]['avg_feature'] = None
                                if(self.candidates[f_indexes[f_dist]]['out_status'] and self.candidates[f_indexes[f_dist]]['in_count'] is None):
                                    self.candidates[f_indexes[f_dist]
                                                    ]['in_count'] = 1
                                    img = Image.open(frames[i].file)
                                    img.save(
                                        "extract_person/IN_{}.jpg".format(self.time+self.candidates[f_indexes[f_dist]]['id']))
                                    SingleCameraTracker.COUNT_IN += 1
                                else:
                                    self.candidates[f_indexes[f_dist]
                                                    ]['in_status'] = True
                                    c_in_temp = -1
                            else:
                                for track2 in self.candidates:
                                    if (self.tracks[-1]['timestamps'][0] > track2['timestamps'][-1]
                                            or track2['timestamps'][0] > self.tracks[-1]['timestamps'][-1]):
                                        if(self._giou(self.tracks[-1]['boxes'][-1], track2['boxes'][-1]) > self.track_detection_iou_thresh):
                                            track2['boxes'].append(
                                                self.tracks[-1]['boxes'][-1])
                                            track2['timestamps'].append(
                                                self.tracks[-1]['timestamps'][-1])
                                            track2['features'].append(
                                                self.tracks[-1]['features'][-1])
                                            if self.tracks[-1]['features'][-1] is not None:
                                                track2['f_cluster'].update(
                                                    features[i])
                                                if track2['avg_feature'] is None:
                                                    track2['avg_feature'] = np.zeros(
                                                        self.tracks[-1]['features'][-1].shape)
                                                track2['avg_feature'] += (self.tracks[-1]['features'][-1] - track2['avg_feature']) / \
                                                    len(self.tracks[-1]
                                                        ['features'])
                                            else:
                                                track2['avg_feature'] = None
                                            track2['in_status'] = True
                                            c_in_temp = -1
                                            break
                        if(c_in_temp == SingleCameraTracker.COUNT_IN):
                            self.candidates.append(
                                copy(self.tracks[-1]))
                if(current_point.within(self.out_poly)):
                    if(self.tracks[-1]['out_status'] == False):
                        self.tracks[-1]['out_status'] = True
                        c_out_temp = SingleCameraTracker.COUNT_OUT
                        if(len(self.candidates) > 0):
                            list_curr_f = [feature['features'][-1]
                                           for feature in self.candidates]
                            list_avg_f = [feature['avg_feature']
                                          for feature in self.candidates]
                            list_clus_f = [clusters_vec_distance(feature['f_cluster'], self.tracks[-1]['features'][-1])
                                           for feature in self.candidates]
                            distances_f = cdist(self.tracks[-1]['features'][-1].reshape(1, -1),
                                                np.array(list_curr_f).reshape(len(list_curr_f), -1), 'cosine')
                            nearest_idx_f = np.argmin(distances_f)
                            distances_avg_f = cdist(self.tracks[-1]['features'][-1].reshape(1, -1),
                                                    np.array(list_avg_f).reshape(len(list_avg_f), -1), 'cosine')
                            nearest_idx_avg = np.argmin(distances_avg_f)
                            nearest_idx_clus = list_clus_f.index(
                                min(list_clus_f))
                            distances = [
                                distances_f[nearest_idx_f], distances_avg_f[nearest_idx_avg], list_clus_f[nearest_idx_clus]]
                            f_indexes = [nearest_idx_f,
                                         nearest_idx_avg, nearest_idx_clus]
                            f_dist = distances.index(min(distances))
                            if(distances[f_dist] < self.rectify_thresh):
                                self.candidates[f_indexes[f_dist]]['boxes'].append(
                                    self.tracks[-1]['boxes'][-1])
                                self.candidates[f_indexes[f_dist]]['timestamps'].append(
                                    self.tracks[-1]['timestamps'][-1])
                                self.candidates[f_indexes[f_dist]]['features'].append(
                                    self.tracks[-1]['features'][-1])
                                if self.tracks[-1]['features'][-1] is not None:
                                    self.candidates[f_indexes[f_dist]]['f_cluster'].update(
                                        features[i])
                                    if self.candidates[f_indexes[f_dist]]['avg_feature'] is None:
                                        self.candidates[f_indexes[f_dist]]['avg_feature'] = np.zeros(
                                            self.tracks[-1]['features'][-1].shape)
                                    self.candidates[f_indexes[f_dist]]['avg_feature'] += (self.tracks[-1]['features'][-1] - self.candidates[f_indexes[f_dist]]['avg_feature']) / \
                                        len(self.tracks[-1]
                                            ['features'])
                                else:
                                    self.candidates[f_indexes[f_dist]
                                                    ]['avg_feature'] = None
                                if(self.candidates[f_indexes[f_dist]]['out_status'] and self.candidates[f_indexes[f_dist]]['in_count'] is None):
                                    self.candidates[f_indexes[f_dist]
                                                    ]['in_count'] = 1
                                    img = Image.open(frames[i].file)
                                    img.save(
                                        "extract_person/IN_{}.jpg".format(self.time+self.candidates[f_indexes[f_dist]]['id']))
                                    SingleCameraTracker.COUNT_IN += 1
                                else:
                                    self.candidates[f_indexes[f_dist]
                                                    ]['in_status'] = True
                                    c_out_temp = -1
                            else:
                                for track2 in self.candidates:
                                    if (self.tracks[-1]['timestamps'][0] > track2['timestamps'][-1]
                                            or track2['timestamps'][0] > self.tracks[-1]['timestamps'][-1]):
                                        if(self._giou(self.tracks[-1]['boxes'][-1], track2['boxes'][-1]) > self.track_detection_iou_thresh):
                                            track2['boxes'].append(
                                                self.tracks[-1]['boxes'][-1])
                                            track2['timestamps'].append(
                                                self.tracks[-1]['timestamps'][-1])
                                            track2['features'].append(
                                                self.tracks[-1]['features'][-1])
                                            if self.tracks[-1]['features'][-1] is not None:
                                                track2['f_cluster'].update(
                                                    features[i])
                                                if track2['avg_feature'] is None:
                                                    track2['avg_feature'] = np.zeros(
                                                        self.tracks[-1]['features'][-1].shape)
                                                track2['avg_feature'] += (self.tracks[-1]['features'][-1] - track2['avg_feature']) / \
                                                    len(self.tracks[-1]
                                                        ['features'])
                                            else:
                                                track2['avg_feature'] = None
                                            track2['in_status'] = True
                                            c_out_temp = -1
                                            break
                        if(c_out_temp == SingleCameraTracker.COUNT_OUT):
                            self.candidates.append(
                                copy(self.tracks[-1]))

    def _create_tracklet_descr(self, timestamp, rect, id, feature):
        return {'id': id,
                'cam_id': self.id,
                'boxes': [rect],
                'timestamps': [timestamp],
                'features': [feature],
                'in_status': False,
                'out_status': False,
                'in_count': None,
                'out_count': None,
                'in_state': 0,
                'out_state': 0,
                'avg_feature': feature.copy() if feature is not None else None,
                'f_cluster': ClusterFeature(self.n_clusters, feature)}

    def _compute_detections_assignment_cost(self, active_tracks_idx, detections, features):
        affinity_matrix = np.zeros(
            (len(detections), len(active_tracks_idx)), dtype=np.float32)
        for i, idx in enumerate(active_tracks_idx):
            track_box = self.tracks[idx]['boxes'][-1]
            track_avg_feat = self.tracks[idx]['avg_feature']
            for j, d in enumerate(detections):
                iou = 0.5 * self._giou(d, track_box) + 0.5
                try:
                    if track_avg_feat is not None and features[j] is not None and features[j].shape[0] == 2048:
                        # compare with average feature
                        reid_sim_avg = 1 - cosine(track_avg_feat, features[j])
                        # compare with last feature
                        reid_sim_curr = 1 - \
                            cosine(self.tracks[idx]
                                   ['features'][-1], features[j])
                        # compare with cluster ?
                        reid_sim_clust = 1 - \
                            clusters_vec_distance(
                                self.tracks[idx]['f_cluster'], features[j])
                        reid_sim = max(
                            reid_sim_avg, reid_sim_curr, reid_sim_clust)
                    else:
                        reid_sim = 0.5
                except:
                    reid_sim = 0.5
                affinity_matrix[j, i] = iou * reid_sim
        return 1 - affinity_matrix

    @staticmethod
    def _area(box):
        return max(box[2] - box[0], 0) * max(box[3] - box[1], 0)

    def _giou(self, b1, b2, a1=None, a2=None):
        if a1 is None:
            a1 = self._area(b1)
        if a2 is None:
            a2 = self._area(b2)
        intersecion = self._area([max(b1[0], b2[0]), max(b1[1], b2[1]),
                                  min(b1[2], b2[2]), min(b1[3], b2[3])])
        enclosing = self._area([min(b1[0], b2[0]), min(b1[1], b2[1]),
                                max(b1[2], b2[2]), max(b1[3], b2[3])])
        u = a1 + a2 - intersecion
        iou = intersecion / u if u > 0 else 0
        giou = iou - (enclosing - u) / enclosing if enclosing > 0 else -1
        return giou

    def _iou(self, b1, b2, a1=None, a2=None):
        if a1 is None:
            a1 = self._area(b1)
        if a2 is None:
            a2 = self._area(b2)
        intersecion = self._area([max(b1[0], b2[0]), max(b1[1], b2[1]),
                                  min(b1[2], b2[2]), min(b1[3], b2[3])])
        u = a1 + a2 - intersecion
        return intersecion / u if u > 0 else 0

    def _ios(self, b1, b2, a1=None, a2=None):
        if a1 is None:
            # intersection over self
            a1 = self._area(b1)
        if a2 is None:
            a2 = self._area(b2)
        intersecion = self._area([max(b1[0], b2[0]), max(b1[1], b2[1]),
                                  min(b1[2], b2[2]), min(b1[3], b2[3])])
        m = a1
        return intersecion / m if m > 0 else 0

    def _get_embeddings(self, images, mask=None):
        embeddings = np.array([])
        if images:
            with torch.no_grad():
                img = torch.cat(
                    [self.data_transform(Image.open(img.file).convert(
                        'RGB')).unsqueeze(0) for img in images], dim=0).float().to("cuda")
                embeddings = self.reid_model.forward(img)
            return embeddings.cpu().numpy()
        else:
            return embeddings

    def _merge_clustered_features(self, clusters1, clusters2, features1, features2):
        if len(features1) >= len(features2):
            for feature in features2:
                try:
                    if feature is not None and feature.shape[0] == 2048:
                        clusters1.update(feature)
                except:
                    continue
            return clusters1
        else:
            for feature in features1:
                try:
                    if feature is not None and feature.shape[0] == 2048:
                        clusters2.update(feature)
                except:
                    continue
            return clusters2

    def _check_velocity_constraint(self, track, detection):
        try:
            dt = abs(self.time - track['timestamps'][-1])
            avg_size = 0
            for det in [track['boxes'][-1], detection]:
                avg_size += 0.5 * (abs(det[2] - det[0]) + abs(det[3] - det[1]))
            avg_size *= 0.5
            shifts = [abs(x - y)
                      for x, y in zip(track['boxes'][-1], detection)]
            velocity = sum(shifts) / len(shifts) / dt / avg_size
            if velocity > self.max_bbox_velocity:
                return False
        except:
            pass
        return True
