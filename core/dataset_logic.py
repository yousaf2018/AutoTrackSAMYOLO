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
        
        # 1. Scan again to be safe (logic reused from UI scan essentially)
        video_files = sorted([p for p in raw_videos_dir.iterdir() if p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]])
        video_info = {}
        
        if self.progress_signal: self.progress_signal.emit(0, "Scanning files...")

        for v in video_files:
            if self.stop_flag: return
            # Try finding CSV in Run_* subfolders
            csv_path = None
            # Recursively find CSV matching video name
            for root, _, files in os.walk(result_dir):
                target = f"{v.stem}_data.csv"
                if target in files:
                    csv_path = Path(root) / target
                    break
            
            if csv_path:
                try:
                    df = pd.read_csv(csv_path)
                    df_sorted = df.sort_values("Global_Frame_ID").reset_index(drop=True)
                    dets_by_frame = defaultdict(list)
                    for _, row in df_sorted.iterrows():
                        dets_by_frame[int(row["Global_Frame_ID"])].append(row)
                    
                    video_info[v] = { "csv": csv_path, "data": dets_by_frame, "frames": sorted(dets_by_frame.keys()) }
                except: pass

        if not video_info: raise Exception("No matched data found.")

        # 2. Sampling
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

        # 4. Export
        self.create_dirs(dataset_root)
        split_map = {**{p:"train" for p in train_set}, **{p:"val" for p in val_set}, **{p:"test" for p in test_set}}

        processed = 0
        for v_path, info in video_info.items():
            if self.stop_flag: return
            cap = cv2.VideoCapture(str(v_path))
            fw = int(cap.get(3)); fh = int(cap.get(4))
            
            f_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                if (v_path, f_idx) in split_map:
                    split = split_map[(v_path, f_idx)]
                    base = f"{v_path.stem}_f{f_idx:06d}"
                    
                    cv2.imwrite(str(dataset_root / split / "images" / f"{base}.jpg"), frame)
                    
                    with open(dataset_root / split / "labels" / f"{base}.txt", 'w') as f:
                        for r in info['data'][f_idx]:
                            cx, cy = float(r["Centroid_X"]), float(r["Centroid_Y"])
                            xn, yn, wn, hn = self.normalize_bbox(cx, cy, box_size, box_size, fw, fh)
                            f.write(f"0 {self.clamp(xn,0,1):.6f} {self.clamp(yn,0,1):.6f} {self.clamp(wn,0,1):.6f} {self.clamp(hn,0,1):.6f}\n")
                    
                    processed += 1
                    if self.progress_signal: self.progress_signal.emit(int(processed/n*100), f"Processing {split}...")
                f_idx += 1
            cap.release()

        # YAML
        y = {'path': str(dataset_root.absolute()), 'train': 'train/images', 'val': 'val/images', 'test': 'test/images', 'nc': 1, 'names': ['particle']}
        with open(dataset_root/"data.yaml", 'w') as f: yaml.dump(y, f)

    def create_dirs(self, r):
        for s in ['train', 'val', 'test']:
            (r/s/"images").mkdir(parents=True, exist_ok=True)
            (r/s/"labels").mkdir(parents=True, exist_ok=True)