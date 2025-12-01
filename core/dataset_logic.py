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

    def normalize_bbox(self, cx, cy, w, h, frame_w, frame_h):
        return (cx / frame_w, cy / frame_h, w / frame_w, h / frame_h)

    def run(self):
        raw_videos_dir = Path(self.config['raw_video_dir'])
        result_dir = Path(self.config['sam_results_dir'])
        dataset_root = Path(self.config['output_dataset_dir'])
        box_size = self.config.get('box_size', 30)
        
        # 1. Collect Info
        self.log("Scanning files...")
        video_files = sorted([p for p in raw_videos_dir.iterdir() if p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]])
        
        if not video_files:
            raise Exception("No videos found in Raw Video Directory.")

        video_info = {}
        
        for v in video_files:
            if self.stop_flag: return
            
            # Look for folder matching video name
            sub = result_dir / v.stem
            # Also try checking if folder starts with Run_ and contains the csv
            if not sub.exists():
                # Search for any folder inside result_dir containing {v.stem}_data.csv
                found = False
                for run_folder in result_dir.glob("Run_*"):
                    if (run_folder / f"{v.stem}_data.csv").exists():
                        sub = run_folder
                        found = True
                        break
                if not found:
                    self.log(f"Skipping {v.name}: No result folder/CSV found.")
                    continue

            csv_files = list(sub.glob("*_data.csv"))
            if not csv_files: continue
            csv_path = csv_files[0]

            try:
                df = pd.read_csv(csv_path)
                df_sorted = df.sort_values("Global_Frame_ID").reset_index(drop=True)
                
                dets_by_frame = defaultdict(list)
                for _, row in df_sorted.iterrows():
                    fid = int(row["Global_Frame_ID"])
                    dets_by_frame[fid].append(row)
                
                video_info[v] = {
                    "csv_path": csv_path,
                    "dets_by_frame": dets_by_frame,
                    "frames": sorted(dets_by_frame.keys())
                }
                self.log(f"Loaded {v.name}: {len(dets_by_frame)} annotated frames.")
                
            except Exception as e:
                self.log(f"Error reading CSV for {v.name}: {e}")

        # 2. Sampling
        all_pairs = []
        for v_path, info in video_info.items():
            for fid in info["frames"]:
                all_pairs.append((v_path, fid))

        total_frames = len(all_pairs)
        self.log(f"Total detected frames available: {total_frames}")
        
        if total_frames == 0:
            raise Exception("No detections found in any CSV files.")

        rng = random.Random(self.config['seed'])
        rng.shuffle(all_pairs)
        
        max_f = self.config['max_frames']
        sampled = all_pairs[:max_f] if max_f < total_frames else all_pairs
        
        # 3. Split
        n = len(sampled)
        n_train = int(n * self.config['train_ratio'])
        n_val = int(n * self.config['val_ratio'])
        
        train_set = set(sampled[:n_train])
        val_set = set(sampled[n_train:n_train+n_val])
        test_set = set(sampled[n_train+n_val:])
        
        self.log(f"Split: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

        # 4. Export
        self.create_dirs(dataset_root)
        
        # Map for O(1) lookup
        split_map = {}
        for p in train_set: split_map[p] = "train"
        for p in val_set: split_map[p] = "val"
        for p in test_set: split_map[p] = "test"

        processed_count = 0
        
        for v_path, info in video_info.items():
            if self.stop_flag: return
            self.log(f"Processing video: {v_path.name}")
            
            cap = cv2.VideoCapture(str(v_path))
            if not cap.isOpened(): continue
            
            fw = int(cap.get(3))
            fh = int(cap.get(4))
            total_v_frames = int(cap.get(7))
            
            # Optional: Annotated video output
            # result_sub = info["csv_path"].parent
            # out_vid = cv2.VideoWriter(str(result_sub / "yolo_debug.mp4"), 
            #                           cv2.VideoWriter_fourcc(*'mp4v'), 30, (fw, fh))

            dets_map = info['dets_by_frame']
            f_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                pair_key = (v_path, f_idx)
                
                # Only save if in sampled set
                if pair_key in split_map:
                    split_name = split_map[pair_key]
                    
                    # Filenames
                    base_name = f"{v_path.stem}_f{f_idx:06d}"
                    img_out = dataset_root / split_name / "images" / f"{base_name}.jpg"
                    lbl_out = dataset_root / split_name / "labels" / f"{base_name}.txt"
                    
                    cv2.imwrite(str(img_out), frame)
                    
                    # Write Labels
                    with open(lbl_out, 'w') as lf:
                        dets = dets_map.get(f_idx, [])
                        for r in dets:
                            cx, cy = float(r["Centroid_X"]), float(r["Centroid_Y"])
                            xn, yn, wn, hn = self.normalize_bbox(cx, cy, box_size, box_size, fw, fh)
                            # Class 0 for object
                            lf.write(f"0 {self.clamp(xn,0,1):.6f} {self.clamp(yn,0,1):.6f} {self.clamp(wn,0,1):.6f} {self.clamp(hn,0,1):.6f}\n")
                    
                    processed_count += 1
                    if self.progress_signal:
                        pct = int((processed_count / n) * 100)
                        self.progress_signal.emit(pct, f"Exporting {split_name}...")

                f_idx += 1
                
            cap.release()
        
        # Write YAML
        yaml_content = {
            'path': str(dataset_root.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,
            'names': ['particle']
        }
        
        with open(dataset_root / "data.yaml", 'w') as f:
            yaml.dump(yaml_content, f)
            
        self.log(f"Success! Dataset created at {dataset_root}")

    def create_dirs(self, root):
        for s in ['train', 'val', 'test']:
            (root / s / "images").mkdir(parents=True, exist_ok=True)
            (root / s / "labels").mkdir(parents=True, exist_ok=True)