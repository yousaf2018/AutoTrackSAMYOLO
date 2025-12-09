import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import random
import shutil
import yaml

class YoloDatasetGenerator:
    def __init__(self, config, log_signal=None, progress_signal=None):
        self.config = config
        self.log_signal = log_signal
        self.progress_signal = progress_signal
        self.stop_flag = False

    def log(self, msg):
        if self.log_signal: self.log_signal.emit(msg)
        else: print(msg)

    def clamp(self, v, lo, hi):
        return max(lo, min(hi, v))

    def run(self):
        raw_videos_dir = Path(self.config['raw_video_dir'])
        result_dir = Path(self.config['sam_results_dir'])
        dataset_root = Path(self.config['output_dataset_dir'])
        
        dataset_type = self.config.get('dataset_type', 'Detection') # 'Detection' or 'Segmentation'
        class_names = self.config.get('class_names', ['object'])
        
        self.log(f"Starting {dataset_type} Dataset Generation...")
        self.log(f"Classes: {class_names}")

        # 1. Scan and Match Files
        video_files = sorted([p for p in raw_videos_dir.iterdir() if p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]])
        video_info = {}
        
        if self.progress_signal: self.progress_signal.emit(0, "Scanning files...")

        for v in video_files:
            if self.stop_flag: return
            
            # Find corresponding CSV
            csv_path = None
            for root, _, files in os.walk(result_dir):
                target = f"{v.stem}_data.csv"
                if target in files:
                    csv_path = Path(root) / target
                    break
            
            if csv_path:
                try:
                    df = pd.read_csv(csv_path)
                    # Group data by frame
                    dets_by_frame = defaultdict(list)
                    for _, row in df.iterrows():
                        dets_by_frame[int(row["Global_Frame_ID"])].append(row)
                    
                    video_info[v] = { "csv": csv_path, "data": dets_by_frame, "frames": sorted(dets_by_frame.keys()) }
                except Exception as e: 
                    self.log(f"Error reading {csv_path}: {e}")

        if not video_info: raise Exception("No matched Video/CSV pairs found.")

        # 2. Sampling Logic
        all_pairs = []
        for v_path, info in video_info.items():
            for fid in info["frames"]:
                all_pairs.append((v_path, fid))

        rng = random.Random(self.config['seed'])
        rng.shuffle(all_pairs)
        
        max_f = self.config['max_frames']
        sampled = all_pairs[:max_f] if max_f < len(all_pairs) else all_pairs
        
        # 3. Split
        n = len(sampled)
        n_train = int(n * self.config['train_ratio'])
        n_val = int(n * self.config['val_ratio'])
        
        train_set = set(sampled[:n_train])
        val_set = set(sampled[n_train:n_train+n_val])
        test_set = set(sampled[n_train+n_val:])

        # 4. Generate Dataset
        self.create_dirs(dataset_root)
        split_map = {**{p:"train" for p in train_set}, **{p:"val" for p in val_set}, **{p:"test" for p in test_set}}

        processed = 0
        total_to_process = len(sampled)

        for v_path, info in video_info.items():
            if self.stop_flag: return
            
            cap = cv2.VideoCapture(str(v_path))
            fw = int(cap.get(3)); fh = int(cap.get(4))
            
            f_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # If this frame is in our random sample
                if (v_path, f_idx) in split_map:
                    split = split_map[(v_path, f_idx)]
                    base_name = f"{v_path.stem}_f{f_idx:06d}"
                    
                    # Save Image
                    img_out = dataset_root / split / "images" / f"{base_name}.jpg"
                    cv2.imwrite(str(img_out), frame)
                    
                    # Save Labels
                    lbl_out = dataset_root / split / "labels" / f"{base_name}.txt"
                    
                    with open(lbl_out, 'w') as f:
                        rows = info['data'][f_idx]
                        for r in rows:
                            # Default class 0, or logic to map Object_ID to class if needed
                            # Here we assume all tracked objects are the same class 0
                            class_id = 0 
                            
                            poly_str = str(r.get("Polygon_Coords", ""))
                            
                            if dataset_type == "Segmentation":
                                label_line = self.format_segmentation(poly_str, class_id, fw, fh)
                            else:
                                # Detection: Calculate BBox from Polygon or Centroid
                                label_line = self.format_detection(r, poly_str, class_id, fw, fh)
                            
                            if label_line:
                                f.write(label_line + "\n")
                    
                    processed += 1
                    if self.progress_signal: 
                        self.progress_signal.emit(int(processed/total_to_process*100), f"Generating {split} set...")
                
                f_idx += 1
            cap.release()

        # 5. Write data.yaml
        yaml_content = {
            'path': str(dataset_root.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(class_names),
            'names': class_names
        }
        with open(dataset_root/"data.yaml", 'w') as f: yaml.dump(yaml_content, f)
            
        self.log(f"Success! {dataset_type} Dataset created at {dataset_root}")

    def format_segmentation(self, poly_str, class_id, fw, fh):
        """Converts pixel polygon string to YOLO normalized polygon line."""
        if not poly_str or poly_str == "nan": return None
        
        points = poly_str.split(';')
        norm_coords = []
        
        for p in points:
            if ',' not in p: continue
            px, py = map(float, p.split(','))
            nx = self.clamp(px / fw, 0.0, 1.0)
            ny = self.clamp(py / fh, 0.0, 1.0)
            norm_coords.extend([f"{nx:.6f}", f"{ny:.6f}"])
            
        if len(norm_coords) < 6: return None # Need at least 3 points
        
        return f"{class_id} " + " ".join(norm_coords)

    def format_detection(self, row, poly_str, class_id, fw, fh):
        """Calculates Bounding Box (XYWH Normalized) from Polygon or Centroid."""
        
        # Try calculating tight box from polygon first
        if poly_str and poly_str != "nan":
            points = poly_str.split(';')
            x_vals = []
            y_vals = []
            for p in points:
                if ',' not in p: continue
                px, py = map(float, p.split(','))
                x_vals.append(px)
                y_vals.append(py)
            
            if x_vals:
                min_x, max_x = min(x_vals), max(x_vals)
                min_y, max_y = min(y_vals), max(y_vals)
                
                width = max_x - min_x
                height = max_y - min_y
                center_x = min_x + (width / 2)
                center_y = min_y + (height / 2)
                
                # Normalize
                n_cx = self.clamp(center_x / fw, 0, 1)
                n_cy = self.clamp(center_y / fh, 0, 1)
                n_w = self.clamp(width / fw, 0, 1)
                n_h = self.clamp(height / fh, 0, 1)
                
                return f"{class_id} {n_cx:.6f} {n_cy:.6f} {n_w:.6f} {n_h:.6f}"

        # Fallback: Fixed Box Size around Centroid
        cx = float(row["Centroid_X"])
        cy = float(row["Centroid_Y"])
        box_size = self.config.get('box_size', 50)
        
        n_cx = self.clamp(cx / fw, 0, 1)
        n_cy = self.clamp(cy / fh, 0, 1)
        n_w = self.clamp(box_size / fw, 0, 1)
        n_h = self.clamp(box_size / fh, 0, 1)
        
        return f"{class_id} {n_cx:.6f} {n_cy:.6f} {n_w:.6f} {n_h:.6f}"

    def create_dirs(self, root):
        for s in ['train', 'val', 'test']:
            (root / s / "images").mkdir(parents=True, exist_ok=True)
            (root / s / "labels").mkdir(parents=True, exist_ok=True)