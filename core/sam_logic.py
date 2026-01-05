import os
import cv2
import torch
import numpy as np
import shutil
import glob
import csv
import gc
import pandas as pd
import matplotlib
# Critical: Use Agg backend to prevent GUI crashes during plotting in threads
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import sys
import time

# Memory optimization for CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- GLOBAL SINGLETON STORAGE ---
# Keeps the model loaded in VRAM between runs to prevent SegFaults
GLOBAL_SAM_MODEL = None
GLOBAL_SAM_PREDICTOR = None

class SAM3Pipeline:
    def __init__(self, output_dir, config, log_signal=None):
        self.output_dir = output_dir
        self.config = config
        self.log_signal = log_signal
        
        # Local references (will point to globals)
        self.model = None
        self.predictor = None
        
        # State
        self.reference_templates = [] 
        self.manual_rects = {} 
        self.current_fps = 0.0
        self.active_objects = 0
        self.stop_flag = False

    def log(self, msg):
        if self.log_signal: self.log_signal.emit(msg)
        else: print(msg)

    def force_gpu_cleanup(self):
        """Clears cache but keeps model loaded."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def cleanup(self):
        """
        Resets pipeline state but KEEPS the model in memory.
        This prevents CUDA context crashes on re-run.
        """
        self.reference_templates = []
        self.manual_rects = {}
        # We do NOT delete self.model or self.predictor here anymore.
        self.force_gpu_cleanup()
        self.log("Resources released (Model kept in VRAM).")

    def load_model(self):
        """Loads SAM3 model once and reuses it for subsequent runs."""
        global GLOBAL_SAM_MODEL, GLOBAL_SAM_PREDICTOR

        if GLOBAL_SAM_MODEL is not None and GLOBAL_SAM_PREDICTOR is not None:
            self.log("♻️ Reusing loaded SAM 3 Model (Skipping reload)...")
            self.model = GLOBAL_SAM_MODEL
            self.predictor = GLOBAL_SAM_PREDICTOR
            return

        self.log("Importing SAM3 modules (First Run)...")
        try:
            import sam3
            from sam3.model_builder import build_sam3_video_model
            
            self.log("Building SAM3 Video Model (loading to GPU)...")
            with torch.inference_mode():
                # Load to Global Variables
                GLOBAL_SAM_MODEL = build_sam3_video_model()
                GLOBAL_SAM_PREDICTOR = GLOBAL_SAM_MODEL.tracker
                GLOBAL_SAM_PREDICTOR.backbone = GLOBAL_SAM_MODEL.detector.backbone
            
            # Assign to local instance
            self.model = GLOBAL_SAM_MODEL
            self.predictor = GLOBAL_SAM_PREDICTOR
            self.log("✅ SAM 3 Model Loaded Successfully.")
            
        except ImportError as e:
            raise ImportError(f"Could not import 'sam3'. Check Path.\nError: {e}")
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")

    def extract_templates_from_rects(self, video_path, templates_dict):
        """
        Extracts templates from ALL annotated frames.
        """
        self.log(f"Extracting templates from {len(templates_dict)} annotated frames...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise Exception("Read failed")
        
        self.reference_templates = []
        self.manual_rects = {} 
        
        sorted_frames = sorted(templates_dict.keys())
        
        for f_idx in sorted_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frame_manuals = []
            for (r, cls_id) in templates_dict[f_idx]:
                x, y, w, h = r.x(), r.y(), r.width(), r.height()
                if w > 2 and h > 2:
                    crop = gray[y:y+h, x:x+w]
                    self.reference_templates.append(crop)
                    # Store tuple (x, y, w, h, class_id)
                    frame_manuals.append((x, y, w, h, cls_id))
            
            if frame_manuals:
                self.manual_rects[f_idx] = frame_manuals
        
        cap.release()
        self.log(f"Extracted {len(self.reference_templates)} templates.")

    def process_video(self, video_path, progress_callback):
        name = os.path.splitext(os.path.basename(video_path))[0]
        out = os.path.join(self.output_dir, name)
        temp = os.path.join(out, "temp")
        proc = os.path.join(out, "proc")
        
        for d in [out, temp, proc]: 
            os.makedirs(d, exist_ok=True)
        
        csv_p = os.path.join(out, f"{name}_data.csv")
        vid_p = os.path.join(out, f"{name}_tracked.mp4")
        
        # CSV Header
        if self.config.get('save_csv', True):
            with open(csv_p, 'w', newline='') as f: 
                csv.writer(f).writerow(["Global_Frame_ID", "Object_ID", "Class_ID", "Centroid_X", "Centroid_Y", "Size_Pixels", "Size_um2", "Polygon_Coords"])

        try:
            # 1. Split
            progress_callback(5, 100, "Splitting...", 0, 0)
            chunk_paths, fps, w, h = self._split_video(video_path, temp)
            if self.stop_flag: return

            # 2. Detect
            progress_callback(10, 100, "Scanning...", 0, 0)
            prompts = self._scan_frame(chunk_paths[0], frame_idx_global=0)
            self.active_objects = len(prompts)
            progress_callback(15, 100, f"Tracking {self.active_objects} objects...", 0, 0)
            
            if not prompts:
                self.log("No objects. Skipped.")
                shutil.rmtree(temp); shutil.rmtree(proc)
                return

            # 3. Track
            total_c = len(chunk_paths)
            f_chunk = int(fps * self.config['chunk_duration'])
            
            for i, cp in enumerate(chunk_paths):
                if self.stop_flag: return
                c_idx = i + 1
                prog = 15 + (i / total_c * 75)
                save_p = os.path.join(proc, f"p_{os.path.basename(cp)}")
                
                def inner_cb(curr_b, tot_b):
                    progress_callback(int(prog), 100, f"Chunk {c_idx}/{total_c}", curr_b, tot_b)

                final_m, next_p = self._track_chunk(cp, save_p, fps, prompts, i, f_chunk, csv_p, inner_cb)
                prompts = next_p
                self.active_objects = len(prompts)
                if not prompts: break

            # 4. Stitch
            if self.config.get('save_video') and not self.config.get('fast_mode'):
                progress_callback(92, 100, "Stitching...", 0, 0)
                self._stitch(proc, vid_p, fps, w, h)

            # 5. Analyze
            progress_callback(96, 100, "Analysis...", 0, 0)
            self._analyze(csv_p, out, name, w, h)
            
            if not self.config.get('save_csv', True) and os.path.exists(csv_p): 
                os.remove(csv_p)
            
            # 6. Clean
            shutil.rmtree(temp)
            shutil.rmtree(proc)
            progress_callback(100, 100, "Done", 0, 0)

        except Exception as e:
            self.log(f"Err: {e}")
            raise e

    # ==========================================================
    # INTERNAL HELPERS
    # ==========================================================

    def _split_video(self, path, out_dir):
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        per_chunk = int(fps * self.config['chunk_duration'])
        
        paths = []; idx = 0
        while True:
            if self.stop_flag: break
            out_file = os.path.join(out_dir, f"c_{idx:03d}.mp4")
            writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            cnt = 0
            while cnt < per_chunk:
                ret, frame = cap.read()
                if not ret: break
                writer.write(frame)
                cnt += 1
            writer.release()
            if cnt > 0: paths.append(out_file)
            else: os.remove(out_file); break
            if not ret: break
            idx += 1
        cap.release()
        return paths, fps, w, h

    def _scan_frame(self, chunk_path, frame_idx_global):
        """Hybrid Scan: Manual + Auto."""
        cap = cv2.VideoCapture(chunk_path); ret, frame = cap.read(); cap.release()
        if not ret: return {}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        detected = {}; oid = 1
        
        # 1. Manual Rects
        # Only inject manual boxes if they are defined for this specific frame
        # (Usually frame 0 is used for propagation start)
        if frame_idx_global in self.manual_rects:
             for (mx, my, mw, mh, cls) in self.manual_rects[frame_idx_global]:
                detected[oid] = {'pt': (mx + mw//2, my + mh//2), 'cls': cls}
                oid += 1
        
        # Also inject frame 0 manuals if we are at the very start
        elif 0 in self.manual_rects and frame_idx_global == 0:
             for (mx, my, mw, mh, cls) in self.manual_rects[0]:
                detected[oid] = {'pt': (mx + mw//2, my + mh//2), 'cls': cls}
                oid += 1
        
        # 2. Return if Manual Mode
        if self.config.get("manual_mode", False):
            return {i: {'pt': (d['pt'][0]/w, d['pt'][1]/h), 'cls': d['cls']} for i, d in detected.items()}

        # 3. Auto Detect
        thresh = self.config['match_threshold']
        for tmpl in self.reference_templates:
            res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= thresh)
            th_t, tw_t = tmpl.shape
            min_d = min(th_t, tw_t) / 1.5
            
            for pt in zip(*loc[::-1]):
                cx = pt[0]+tw_t//2
                cy = pt[1]+th_t//2
                new = True
                for _, d in detected.items():
                    ex, ey = d['pt']
                    if np.sqrt((cx-ex)**2+(cy-ey)**2) < min_d: 
                        new = False; break
                
                if new:
                    # Auto-detected objects get Class 0 (Default)
                    detected[oid] = {'pt': (cx, cy), 'cls': 0}
                    oid += 1
                
        return {i: {'pt': (d['pt'][0]/w, d['pt'][1]/h), 'cls': d['cls']} for i, d in detected.items()}

    def _track_chunk(self, chunk_path, save_path, fps, prompts, chunk_idx, total_frames_in_chunk, csv_path, cb):
        batch_size = self.config['batch_size']
        px_scale = self.config['pixel_scale_um'] ** 2
        save_csv = self.config.get('save_csv', True)
        fast_mode = self.config.get('fast_mode', False)
        should_save_video = self.config.get('save_video', False)
        
        all_ids = list(prompts.keys())
        vis_data = {}
        last_masks = {}
        csv_buffer = []
        
        f_csv = open(csv_path, 'a', newline='')
        writer = csv.writer(f_csv) if save_csv else None
        
        total_batches = (len(all_ids) + batch_size - 1) // batch_size
        
        try:
            for i in range(0, len(all_ids), batch_size):
                if self.stop_flag: break
                t_start = time.time()
                current_batch = (i // batch_size) + 1
                cb(current_batch, total_batches)
                
                batch = all_ids[i : i+batch_size]
                self.force_gpu_cleanup()
                
                # Reuse global model predictor
                state = self.predictor.init_state(video_path=chunk_path)
                
                for oid in batch:
                    p_data = prompts[oid]
                    pt_norm = p_data['pt']
                    self.predictor.add_new_points(state, 0, oid, torch.tensor([[pt_norm[0], pt_norm[1]]], dtype=torch.float32), torch.tensor([1], dtype=torch.int32))
                
                frames_processed = 0
                for f_idx, oids, _, masks, _ in self.predictor.propagate_in_video(
                    state, start_frame_idx=0, max_frame_num_to_track=200, reverse=False, propagate_preflight=True
                ):
                    if f_idx not in vis_data: vis_data[f_idx] = {}
                    
                    for j, oid in enumerate(oids):
                        m = (masks[j] > 0.0).cpu().numpy().squeeze()
                        c, area = self._get_centroid_area(m)
                        if c:
                            poly_str = self._get_polygon_str(m)
                            cls = prompts[oid]['cls']
                            
                            if writer:
                                g_frame = (chunk_idx * total_frames_in_chunk) + f_idx
                                csv_buffer.append([g_frame, oid, cls, c[0], c[1], area, area * px_scale, poly_str])
                            
                            if should_save_video and not fast_mode:
                                vis_data[f_idx][oid] = {'pt': c, 'cls': cls}
                            
                            last_masks[oid] = m
                    frames_processed += 1
                
                self.predictor.clear_all_points_in_video(state)
                del state
                self.force_gpu_cleanup()
                
                t_end = time.time()
                duration = t_end - t_start
                if duration > 0: self.current_fps = frames_processed / duration

        finally:
            if writer:
                writer.writerows(csv_buffer)
                f_csv.close()
            
        if should_save_video and not fast_mode:
            self._render(chunk_path, save_path, fps, vis_data, all_ids)
            
        next_prompts = {}
        cap = cv2.VideoCapture(chunk_path)
        w = int(cap.get(3)); h = int(cap.get(4))
        cap.release()
        
        for oid in prompts:
            if oid in last_masks:
                c, _ = self._get_centroid_area(last_masks[oid])
                if c: next_prompts[oid] = {'pt': (c[0]/w, c[1]/h), 'cls': prompts[oid]['cls']}
        
        return last_masks, next_prompts

    def _render(self, inp, out, fps, data, ids):
        cap = cv2.VideoCapture(inp)
        w = int(cap.get(3)); h = int(cap.get(4))
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        # Consistent colors for Class IDs (not Object IDs)
        # Class 0=Green, 1=Red, 2=Blue, etc.
        class_colors = [
            (20, 255, 57),   # Green
            (50, 50, 255),   # Red
            (255, 100, 50),  # Blue
            (0, 255, 255),   # Yellow
            (255, 0, 255)    # Magenta
        ]
        
        curr_frame = 0
        while True:
            if self.stop_flag: break
            ret, frame = cap.read()
            if not ret: break
            
            if curr_frame in data:
                for oid, info in data[curr_frame].items():
                    center = info['pt']
                    cls = info['cls']
                    color = class_colors[cls % len(class_colors)]
                    
                    cv2.circle(frame, center, 4, color, -1)
                    cv2.putText(frame, f"ID:{oid} C:{cls}", (center[0]+6, center[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            writer.write(frame)
            curr_frame += 1
            
        cap.release()
        writer.release()

    def _stitch(self, chunk_dir, output_path, fps, w, h):
        files = sorted(glob.glob(os.path.join(chunk_dir, "*.mp4")))
        if not files: return
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for f in files:
            if self.stop_flag: break
            cap = cv2.VideoCapture(f)
            while True:
                ret, frame = cap.read()
                if not ret: break
                writer.write(frame)
            cap.release()
        writer.release()

    def _analyze(self, csv_path, out_dir, name, w, h):
        try:
            if not os.path.exists(csv_path): return
            df = pd.read_csv(csv_path)
            if df.empty: return
            
            if self.config.get('save_hist', True):
                stats = df.groupby('Object_ID')['Size_um2'].mean()
                plt.figure()
                plt.hist(stats, bins=20, color='skyblue', edgecolor='black')
                plt.title(f"Size Distribution (n={len(stats)})")
                plt.xlabel("Area (um^2)"); plt.ylabel("Count")
                plt.savefig(os.path.join(out_dir, f"{name}_hist.png"))
                plt.close()
            
            if self.config.get('save_heatmap', True):
                plt.figure()
                plt.hist2d(df['Centroid_X'], df['Centroid_Y'], bins=[50, 50], range=[[0, w], [0, h]], cmap='inferno')
                plt.colorbar(); plt.gca().invert_yaxis()
                plt.savefig(os.path.join(out_dir, f"{name}_heat.png"))
                plt.close()

            if self.config.get('save_traj', True):
                plt.figure(); plt.gca().invert_yaxis()
                unique_ids = df['Object_ID'].unique()
                if len(unique_ids) > 200: np.random.shuffle(unique_ids); unique_ids=unique_ids[:200]
                for oid in unique_ids:
                    track = df[df['Object_ID'] == oid]
                    plt.plot(track['Centroid_X'], track['Centroid_Y'], linewidth=0.5, alpha=0.6)
                plt.xlim(0, w); plt.ylim(h, 0)
                plt.savefig(os.path.join(out_dir, f"{name}_traj.png"))
                plt.close()

        except Exception as e:
            self.log(f"Analysis Error: {e}")

    def _get_centroid_area(self, mask):
        if mask is None: return None, 0
        a = np.count_nonzero(mask)
        rows, cols = np.where(mask)
        if len(rows) == 0: return None, 0
        return (int(np.mean(cols)), int(np.mean(rows))), a

    def _get_polygon_str(self, mask):
        if mask is None: return ""
        mask_uint8 = (mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return ""
        cnt = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        return ";".join([f"{p[0][0]},{p[0][1]}" for p in approx])