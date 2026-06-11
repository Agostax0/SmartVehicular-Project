import carla
import queue
import numpy as np
import math
import csv
import typer
import matplotlib.pyplot as plt
import os

from core.engine import SimulationEngine
from core.config import load_config
from scenarios import get_scenario
from utils.carla_utils import image_to_bgr

class EvaluatorEngine(SimulationEngine):
    def run_eval(self, max_frames=100, output_csv="evaluation_results.csv"):
        image_queue = queue.Queue()
        depth_queue = queue.Queue()
        
        self.depth_camera.listen(depth_queue.put)
        self.camera.listen(image_queue.put)
        
        results_data = []
        
        print(f"Starting evaluation for {max_frames} frames...")
        
        try:
            for frame_idx in range(max_frames):
                self.world.tick()
                
                try:
                    image = image_queue.get(timeout=2.0)
                    depth_image = depth_queue.get(timeout=2.0)
                except queue.Empty:
                    continue

                self.state.depth_array = self.perception.process_depth(depth_image)

                current_vel = self.vehicle.get_velocity()
                self.state.ego_speed = np.sqrt(current_vel.x**2 + current_vel.y**2 + current_vel.z**2)

                bgr_array = image_to_bgr(image)
                rgb_array = bgr_array[:, :, ::-1]
                
                self.state.results = self.detector.detect(rgb_array)
                self.perception.analyze_detections(self.state, self.scenario)
                self.state.frame = rgb_array

                # ----------------- Evaluation Metrics -----------------
                # Get ground truth
                if self.walker and self.vehicle:
                    ped_loc = self.walker.get_location()
                    veh_trans = self.vehicle.get_transform()
                    
                    dx = ped_loc.x - veh_trans.location.x
                    dy = ped_loc.y - veh_trans.location.y
                    yaw = math.radians(veh_trans.rotation.yaw)
                    
                    # True forward (Z) and lateral (X) in ego frame
                    true_z = dx * math.cos(yaw) + dy * math.sin(yaw)
                    true_x = -dx * math.sin(yaw) + dy * math.cos(yaw)
                    
                    # True relative velocity
                    ped_vel = self.walker.get_velocity()
                    veh_vel = self.vehicle.get_velocity()
                    rel_vx = ped_vel.x - veh_vel.x
                    rel_vy = ped_vel.y - veh_vel.y
                    
                    true_vz = rel_vx * math.cos(yaw) + rel_vy * math.sin(yaw)
                    true_vx = -rel_vx * math.sin(yaw) + rel_vy * math.cos(yaw)
                    
                    true_ttc = true_z / abs(true_vz) if true_vz < -0.1 else 999.0
                    
                    # Get estimated metrics from state
                    est_z = -1
                    est_x = -1
                    est_ttc = 999.0
                    if self.state.kalman_filters:
                        # Assuming one tracked object
                        for obj_id, kf in self.state.kalman_filters.items():
                            if kf.initialized:
                                # kf.kf.statePost is [X, Z, Vx, Vz]
                                state_post = kf.kf.statePost
                                est_x = float(state_post[0][0])
                                est_z = float(state_post[1][0])
                                vel_z = float(state_post[3][0])
                                real_v = vel_z - self.state.ego_speed
                                if real_v < -0.1:
                                    est_ttc = est_z / abs(real_v)
                            break
                    
                    results_data.append({
                        "frame": frame_idx,
                        "true_z": true_z,
                        "est_z": est_z,
                        "true_x": true_x,
                        "est_x": est_x,
                        "true_ttc": true_ttc,
                        "est_ttc": est_ttc,
                        "brake_needed": self.state.brake_needed
                    })

                # Ego update (keep constant speed for clean eval)
                if self.state.brake_needed:
                    self.controller.apply_control(throttle=0.0, steer=0.0, brake=1.0)
                else:
                    self.controller.apply_throttle(self.scenario.target_speed_mps)
                
                # Visualizer
                if self.state.frame is not None:
                    running, _ = self.visualizer.update(self.state.frame, self.state.results, self.state.kalman_predictions)
                    if not running:
                        break
                        
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
            
        # Write to CSV
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "true_z", "est_z", "true_x", "est_x", "true_ttc", "est_ttc", "brake_needed"])
            writer.writeheader()
            writer.writerows(results_data)
        
        print(f"Saved evaluation results to {output_csv}")
        self.plot_results(output_csv)

    def plot_results(self, csv_file):
        frames, true_z, est_z, true_x, est_x, true_ttc, est_ttc, brakes = [], [], [], [], [], [], [], []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frames.append(int(row["frame"]))
                true_z.append(float(row["true_z"]))
                # Filter out undetected frames for cleaner plots
                ez = float(row["est_z"])
                est_z.append(ez if ez != -1 else None)
                true_x.append(float(row["true_x"]))
                ex = float(row["est_x"])
                est_x.append(ex if ex != -1 else None)
                true_ttc.append(float(row["true_ttc"]))
                et = float(row["est_ttc"])
                est_ttc.append(et if et != 999.0 else None)
                brakes.append(1 if row["brake_needed"] == 'True' else 0)
                
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        # 1. Depth (Z) Tracking
        axs[0].plot(frames, true_z, 'g-', label='True Depth (Z)')
        axs[0].plot(frames, est_z, 'r--', label='Estimated Depth')
        axs[0].set_title('Distance Estimation Quality')
        axs[0].set_ylabel('Meters')
        axs[0].legend()
        axs[0].grid(True)
        
        # 2. Lateral (X) Tracking
        axs[1].plot(frames, true_x, 'b-', label='True Lateral (X)')
        axs[1].plot(frames, est_x, 'm--', label='Estimated Lateral')
        axs[1].set_title('Lateral Position Estimation Quality')
        axs[1].set_ylabel('Meters')
        axs[1].legend()
        axs[1].grid(True)
        
        # 3. TTC and Braking
        axs[2].plot(frames, true_ttc, 'c-', label='True TTC')
        axs[2].plot(frames, est_ttc, 'orange', linestyle='--', label='Estimated TTC')
        # Scale brake signal to make it visible on the same plot (e.g., 0 to 5)
        axs[2].fill_between(frames, 0, [b * 5 for b in brakes], color='red', alpha=0.2, label='AEB Active')
        axs[2].set_title('Time-to-Collision (TTC) & Braking Response')
        axs[2].set_xlabel('Frame')
        axs[2].set_ylabel('Seconds')
        axs[2].set_ylim(0, 10) # focus on critical 10s window
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout()
        plot_file = csv_file.replace('.csv', '.png')
        plt.savefig(plot_file)
        print(f"Saved evaluation plots to {plot_file}")

def main(config_file: str = "config/config.yaml", scenario: str = "colliding-pedestrian", frames: int = 150):
    config_data = load_config(config_file)
    scenario_config = get_scenario(config_data, scenario)
    evaluator = EvaluatorEngine(config_data, scenario_config)
    evaluator.setup()
    evaluator.run_eval(max_frames=frames)

if __name__ == "__main__":
    typer.run(main)
