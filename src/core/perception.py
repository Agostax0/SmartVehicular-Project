import numpy as np
from utils.kalman import TrajectoryKalmanFilter
from utils.collision_module import check_collision_risk

class PerceptionSystem:
    def __init__(self, config):
        self.fps = config["simulation"].get("fps", 20)
        self.img_w = config["sensors"]["camera"]["width"]
        self.img_h = config["sensors"]["camera"]["height"]
        fov_rad = np.deg2rad(config["sensors"]["camera"]["fov"])
        self.f_length = (self.img_w / 2.0) / np.tan(fov_rad / 2.0)

    def process_depth(self, depth_image):
        array = np.frombuffer(depth_image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (depth_image.height, depth_image.width, 4))
        B = array[:, :, 0].astype(np.float32)
        G = array[:, :, 1].astype(np.float32)
        R = array[:, :, 2].astype(np.float32)
        normalized_depth = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
        return 1000.0 * normalized_depth

    def _box_depth(self, depth_array, x1, y1, x2, y2, cx, cy):
        cx_int = max(0, min(self.img_w - 1, int(cx)))
        cy_int = max(0, min(self.img_h - 1, int(cy)))
        fallback = float(depth_array[cy_int, cx_int])

        roi_w = max(2, int((x2 - x1) * 0.35))
        roi_h = max(2, int((y2 - y1) * 0.35))
        left = max(0, cx_int - roi_w // 2)
        right = min(self.img_w, cx_int + roi_w // 2 + 1)
        top = max(0, cy_int - roi_h // 2)
        bottom = min(self.img_h, cy_int + roi_h // 2 + 1)

        roi = depth_array[top:bottom, left:right]
        valid = roi[np.isfinite(roi) & (roi > 0.0)]
        if valid.size:
            return float(np.median(valid))
        return fallback if fallback > 0.0 else 10.0

    def analyze_detections(self, state, scenario):
        predictions = {}
        state.brake_needed = False
        
        if state.results and state.results.boxes and state.results.boxes.id is not None:
            for i, box in enumerate(state.results.boxes):
                obj_id = int(state.results.boxes.id[i])
                coords = box.xyxy[0].tolist()
                
                x1, y1, x2, y2 = coords
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1
                
                Z = 10.0 # fallback
                if state.depth_array is not None:
                    Z = self._box_depth(state.depth_array, x1, y1, x2, y2, cx, cy)
                    
                X = (cx - self.img_w / 2.0) * Z / self.f_length
                
                if obj_id not in state.kalman_filters:
                    state.kalman_filters[obj_id] = TrajectoryKalmanFilter(dt=1.0/self.fps)
                
                est_x, est_z, vel_x, vel_z = state.kalman_filters[obj_id].predict_update(
                    X, Z, ego_vx=state.ego_vx, ego_vz=state.ego_vz,
                    dt=getattr(state, "sensor_dt", None)
                )

                # Relative velocities in the camera frame (pedestrian minus ego).
                # The filter returns ABSOLUTE pedestrian velocities; the ego moves
                # too, so the closing rate on Z is vel_z - ego_vz (= real_v, also
                # used for TTC) and the lateral drift is vel_x - ego_vx. Extrapolat-
                # ing relative positions with absolute velocities was the bug that
                # made predictions stay "stuck" far away at high ego speed.
                rel_vx = vel_x - state.ego_vx
                rel_vz = vel_z - state.ego_vz

                risk, msg, ttc = check_collision_risk(est_x, est_z, rel_vx, rel_vz)
                if risk:
                    state.brake_needed = True

                future_frames = scenario.prediction_future_frames
                kf = state.kalman_filters[obj_id]
                dt = kf.dt

                if getattr(kf, "update_count", 0) < 3:
                    continue

                pred_X = est_x + rel_vx * dt * future_frames
                pred_Z = est_z + rel_vz * dt * future_frames

                if pred_Z <= 1.0:
                    continue

                pred_Z_safe = pred_Z

                pred_cx = (pred_X * self.f_length) / pred_Z_safe + self.img_w / 2.0

                scale_factor = Z / pred_Z_safe
                pred_cy = self.img_h / 2.0 + (cy - self.img_h / 2.0) * scale_factor
                pred_w = w * scale_factor
                pred_h = h * scale_factor

                pred_box = [
                    pred_cx - pred_w / 2.0,
                    pred_cy - pred_h / 2.0,
                    pred_cx + pred_w / 2.0,
                    pred_cy + pred_h / 2.0
                ]

                trajectory = []
                for frm in range(2, future_frames + 1, 4):
                    pt_X = est_x + rel_vx * dt * frm
                    pt_Z = est_z + rel_vz * dt * frm

                    if pt_Z > 0.5:
                        pt_cx = (pt_X * self.f_length) / pt_Z + self.img_w / 2.0
                        pt_cy = self.img_h / 2.0 + (cy - self.img_h / 2.0) * (Z / pt_Z)
                        trajectory.append((pt_cx, pt_cy))
                
                predictions[obj_id] = {
                    "box": pred_box,
                    "trajectory": trajectory,
                    "time_horizon": future_frames * dt
                }
                
        state.kalman_predictions = predictions
