import numpy as np
import scipy
from collections import deque

class KalmanFilter:
    def __init__(self):
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        self.motion_mat = np.eye(8, 8, dtype=np.float32)
        for i in range(4):
            self.motion_mat[i, i + 4] = 1
        self.update_mat = np.eye(4, 8, dtype=np.float32)

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos, dtype=np.float32)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self.motion_mat @ mean
        covariance = self.motion_mat @ covariance @ self.motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        mean = self.update_mat @ mean
        covariance = self.update_mat @ covariance @ self.update_mat.T
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        # Solve: projected_cov * K.T = (P * H.T).T = H * P
        # Then transpose to get K
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), self.update_mat @ covariance, check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance

class Track:
    def __init__(self, mean, covariance, track_id, class_id, conf):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.class_id = class_id
        self.conf = conf
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.track_state = 'new'
        self.kalman_filter = KalmanFilter()

    def to_tlwh(self):
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def predict(self):
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection, class_id, conf):
        xyah = self.tlwh_to_xyah(detection)
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, xyah)
        self.hits += 1
        self.time_since_update = 0
        self.track_state = 'tracked'
        self.class_id = class_id
        self.conf = conf

    def mark_missed(self):
        if self.track_state == 'new':
            self.track_state = 'deleted'
        elif self.time_since_update > self.max_age:
            self.track_state = 'deleted'
        else:
            self.track_state = 'lost'

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
class ByteTrack:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        Track.max_age = int(frame_rate / 30.0 * track_buffer)
        self.kalman_filter = KalmanFilter()
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        self.next_track_id = 1  # Global track ID counter

    def update(self, dets, classes, confs):
        self.frame_id += 1
        activated_starcks = []
        refind_starcks = []
        lost_starcks = []
        removed_starcks = []
        remain_inds = confs > self.track_thresh
        inds_low = confs > 0.1
        inds_high = confs < self.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = dets[inds_second]
        dets = dets[remain_inds]
        classes_keep = classes[remain_inds]
        confs_keep = confs[remain_inds]
        classes_second = classes[inds_second]
        confs_second = confs[inds_second]
        
        # Remove old lost tracks BEFORE creating strack_pool
        self.lost_tracks = [t for t in self.lost_tracks if self.frame_id - t.time_since_update <= self.track_buffer]
        
        if len(dets) > 0:
            detections = self.tlwh_to_xyah(dets)
            unconfirmed = []
            tracked_stracks = []
            for track in self.tracked_tracks:
                if track.track_state != 'tracked':
                    unconfirmed.append(track)
                else:
                    tracked_stracks.append(track)
            strack_pool, unconfirmed_tracks = self.joint_stracks(tracked_stracks, self.lost_tracks)
            self.predict(strack_pool)
            dists = self.get_dists(strack_pool, detections)
            
            # Reduced debug output
            if self.frame_id % 100 == 0:
                print(f"[ByteTrack] Matching: {len(strack_pool)} tracks vs {len(detections)} detections")
            
            matches, u_track, u_detection = self.linear_assignment(dists, self.match_thresh)
            
            # Debug matching results
            if self.frame_id % 100 == 0:
                print(f"[ByteTrack] matches={len(matches)}, active_tracks={len(self.tracked_tracks)}, lost_tracks={len(self.lost_tracks)}")
                if len(dists) > 0 and dists.size > 0:
                    print(f"[ByteTrack] IoU: min_dist={dists.min():.3f}, max_dist={dists.max():.3f}, thresh={self.match_thresh}")
            
            for itracked, idet in matches:
                track = strack_pool[itracked]
                if track.track_state == 'tracked':
                    track.update(dets[idet], classes_keep[idet], confs_keep[idet])  # Pass tlwh, Track.update converts to xyah
                    activated_starcks.append(track)
                else:
                    track.update(dets[idet], classes_keep[idet], confs_keep[idet])  # Pass tlwh, Track.update converts to xyah
                    refind_starcks.append(track)
            if len(dets_second) > 0:
                detections_second = self.tlwh_to_xyah(dets_second)
                r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].track_state == 'tracked']
                dists = self.get_dists(r_tracked_stracks, detections_second, is_second_round=True)
                matches, u_track_second, u_detection_second = self.linear_assignment(dists, 0.5)
                for itracked, idet in matches:
                    track = r_tracked_stracks[itracked]
                    track.update(dets_second[idet], classes_second[idet], confs_second[idet])  # Pass tlwh, Track.update converts to xyah
                    refind_starcks.append(track)
                
                # Mark unmatched second-round tracks as lost
                for i in u_track_second:
                    track = r_tracked_stracks[i]
                    if not track.track_state == 'lost':
                        track.mark_missed()
                        lost_starcks.append(track)
                
                # Create new tracks for unmatched second-round detections
                for i in u_detection_second:
                    if i < len(detections_second) and i < len(classes_second):
                        mean, covariance = self.kalman_filter.initiate(detections_second[i])
                        track = self.init_track(mean, covariance, classes_second[i], confs_second[i])
                        activated_starcks.append(track)
            
            # Mark remaining unmatched first-round tracks as lost
            for it in u_track:
                track = strack_pool[it]
                if not track.track_state == 'lost':
                    track.mark_missed()
                    lost_starcks.append(track)
            
            # Debug new tracks creation
            if self.frame_id % 100 == 0 and len(u_detection) > 0:
                print(f"[ByteTrack] Creating {len(u_detection)} new tracks")
            
            for i in u_detection:
                # Initialize Kalman filter for new detection
                if i < len(detections) and i < len(classes_keep):
                    mean, covariance = self.kalman_filter.initiate(detections[i])
                    track = self.init_track(mean, covariance, classes_keep[i], confs_keep[i])
                    activated_starcks.append(track)
                else:
                    print(f"[ByteTrack WARNING] Invalid index i={i} for detections len={len(detections)}, classes len={len(classes_keep)}")
        else:
            # NO DETECTIONS - but we still predict existing tracks using Kalman filter
            # This is CRITICAL for maintaining IDs during motion blur / fast movement
            for track in self.tracked_tracks:
                if track.track_state == 'tracked':
                    track.predict()  # Kalman prediction of next position
                    track.mark_missed()  # Mark as temporarily lost
                    lost_starcks.append(track)  # Move to lost (but keep predicting)
        
        for track in self.lost_tracks:
            if self.frame_id - track.time_since_update > self.track_buffer:
                removed_starcks.append(track)
        # Add activated and refound tracks to tracked_tracks
        self.tracked_tracks, self.lost_tracks = self.add_stracks(self.tracked_tracks, activated_starcks + refind_starcks)
        # Filter out non-tracked states AFTER adding new tracks
        self.tracked_tracks = [t for t in self.tracked_tracks if t.track_state == 'tracked']
        self.lost_tracks, removed_starcks = self.add_stracks(self.lost_tracks, lost_starcks, is_lost=True)
        self.removed_tracks.extend(removed_starcks)

        
        # Return all active tracks INCLUDING recently lost ones (with Kalman predictions)
        # This maintains visible boxes with persistent IDs during temporary detection gaps
        output_stracks = [track for track in self.tracked_tracks]
        # Also include lost tracks that are still within buffer window (show predicted positions)
        recent_lost = [track for track in self.lost_tracks 
                      if self.frame_id - track.time_since_update <= 10]  # Show for 10 frames
        output_stracks.extend(recent_lost)
        return output_stracks

    def tlwh_to_xyah(self, tlwh):
        ret = np.asarray(tlwh, dtype=np.float64).copy()
        ret[:, :2] += ret[:, 2:] / 2
        ret[:, 2] /= ret[:, 3]
        return ret
    
    def joint_stracks(self, tlista, tlistb):
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res, [t for t in self.tracked_tracks if t.track_state != 'tracked']
    
    def predict(self, tracks):
        for track in tracks:
            track.predict()
    
    def get_dists(self, tracks, detections, is_second_round=False):
        if is_second_round:
            dists = self.iou_distance(tracks, detections)
        else:
            dists = self.iou_distance(tracks, detections)
        return dists

    def iou_distance(self, atracks, btracks):
        if isinstance(btracks, np.ndarray):
            # btracks are detections in xyah format, need to convert to tlwh
            btracks_ = self.xyah_to_tlwh(btracks)
        else:
            btracks_ = np.array([track.to_tlwh() for track in btracks])
        atlwhs = [track.to_tlwh() for track in atracks]
        if len(atlwhs) == 0 or len(btracks_) == 0:
            # Return proper shape (tracks, detections) even when one is empty
            return np.empty((len(atlwhs), len(btracks_)))
        ious = self.ious(atlwhs, btracks_)
        return 1 - ious
    
    def xyah_to_tlwh(self, xyah):
        """Convert from (center_x, center_y, aspect_ratio, height) to (left, top, width, height)"""
        ret = np.asarray(xyah, dtype=np.float64).copy()
        ret[:, 2] *= ret[:, 3]  # width = aspect_ratio * height
        ret[:, 0] -= ret[:, 2] / 2  # left = center_x - width/2
        ret[:, 1] -= ret[:, 3] / 2  # top = center_y - height/2
        return ret

    def ious(self, atlwhs, btlwhs):
        ious = np.zeros((len(atlwhs), len(btlwhs)))
        if ious.size == 0:
            return ious
        for i, atlwh in enumerate(atlwhs):
            for j, btlwh in enumerate(btlwhs):
                ious[i, j] = self.iou(atlwh, btlwh)
        return ious

    def iou(self, atlwh, btlwh):
        a = [atlwh[0], atlwh[1], atlwh[0] + atlwh[2], atlwh[1] + atlwh[3]]
        b = [btlwh[0], btlwh[1], btlwh[0] + btlwh[2], btlwh[1] + btlwh[3]]
        i = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
        area_i = max(0, i[2] - i[0]) * max(0, i[3] - i[1])
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        iou_val = area_i / (area_a + area_b - area_i) if (area_a + area_b - area_i) > 0 else 0
        return iou_val
    
    def linear_assignment(self, dist_matrix, thresh):
        x, y = scipy.optimize.linear_sum_assignment(dist_matrix)
        matches = np.array(list(zip(x, y)))
        if len(matches) == 0:
            # dist_matrix shape is (tracks, detections), so shape[0]=tracks, shape[1]=detections
            return np.empty((0, 2), dtype=int), np.arange(dist_matrix.shape[0]), np.arange(dist_matrix.shape[1])
        else:
            unmatched_a = []
            for i in range(dist_matrix.shape[1]):
                if i not in matches[:, 1]:
                    unmatched_a.append(i)
            unmatched_b = []
            for i in range(dist_matrix.shape[0]):
                if i not in matches[:, 0]:
                    unmatched_b.append(i)
            matches_ = []
            for i, j in matches:
                if dist_matrix[i, j] < thresh:
                    matches_.append([i, j])
                else:
                    unmatched_a.append(j)
                    unmatched_b.append(i)
            if len(matches_) == 0:
                # Return: matches, u_track (unmatched tracks=b), u_detection (unmatched detections=a)
                return np.empty((0, 2), dtype=int), np.array(unmatched_b), np.array(unmatched_a)
            else:
                return np.array(matches_), np.array(unmatched_b), np.array(unmatched_a)
    
    def init_track(self, mean, covariance, class_id, conf):
        track = Track(mean, covariance, self.next_track_id, class_id, conf)
        self.next_track_id += 1  # Increment for next track
        return track
    
    def add_stracks(self, tracks, new_tracks, is_lost=False):
        if is_lost:
            for track in new_tracks:
                track.mark_missed()
            tracks.extend(new_tracks)
        else:
            for track in new_tracks:
                if track.track_state == 'new':
                    track.track_state = 'tracked'
                    tracks.append(track)
        return tracks, [t for t in self.lost_tracks if t.track_state == 'lost']