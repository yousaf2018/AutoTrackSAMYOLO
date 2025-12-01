import os
import sys
import shutil
import glob
import cv2
import numpy as np
import torch
import traceback
import gc
import csv
import pandas as pd
import matplotlib

# Force stable backend for Linux GUI
try:
    matplotlib.use('TkAgg') 
except:
    pass
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches

# --- MEMORY OPTIMIZATION ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- SAM 3 Imports ---
import sam3
from sam3.model_builder import build_sam3_video_model

# ================= CONFIGURATION =================
# FOLDER CONTAINING YOUR VIDEOS
INPUT_FOLDER = "/mnt/Zebrafish_24TB/Mahmood-Yousaf-Data/nanoparticles/" 

# WHERE TO SAVE RESULTS
MAIN_OUTPUT_DIR = "/mnt/Zebrafish_24TB/Mahmood-Yousaf-Data/20251124-v3-Nanoparticles-SAM3_Batch_Analysis_Results"

# --- PHYSICS & CALIBRATION ---
PIXEL_SCALE_UM = 0.324             # 1 pixel = 0.324 micrometers (324nm)
PIXEL_AREA_CONVERSION = PIXEL_SCALE_UM ** 2 # ~0.105 um^2 per pixel

# --- SYSTEM SETTINGS ---
CHUNK_DURATION = 5                 # Seconds per split
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MATCH_THRESHOLD = 0.85             # Sensitivity (Higher = stricter match)
BATCH_SIZE = 5                     # Keep low (5) for 12GB GPU memory safety
MAX_TRACK_FRAMES = 130             # Buffer size for tracking
# =================================================

def force_gpu_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def setup_environment():
    force_gpu_cleanup()
    if DEVICE == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

def get_random_color(obj_id):
    np.random.seed(obj_id * 19)
    return tuple(np.random.randint(50, 255, 3).tolist())

# ==========================================================
# VIDEO UTILS
# ==========================================================
def split_video(video_path, temp_dir, duration=5):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_chunk = int(fps * duration)
    
    chunk_paths = []
    chunk_idx = 0
    
    print(f"  -> Splitting video into {duration}s chunks...")
    while True:
        output_filename = os.path.join(temp_dir, f"chunk_{chunk_idx:03d}.mp4")
        out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        frames_written = 0
        while frames_written < frames_per_chunk:
            ret, frame = cap.read()
            if not ret: break
            out.write(frame)
            frames_written += 1
        out.release()
        if frames_written > 0:
            chunk_paths.append(output_filename)
            chunk_idx += 1
        if not ret: break
    cap.release()
    return chunk_paths, width, height, fps

def stitch_processed_chunks(chunk_folder, output_path, fps, width, height):
    print(f"  -> Stitching chunks into final video: {output_path}")
    processed_chunks = sorted(glob.glob(os.path.join(chunk_folder, "processed_chunk_*.mp4")))
    if not processed_chunks:
        print("    ! No processed chunks found to stitch.")
        return
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for chunk in processed_chunks:
        cap = cv2.VideoCapture(chunk)
        while True:
            ret, frame = cap.read()
            if not ret: break
            out.write(frame)
        cap.release()
    out.release()
    print("  -> Stitching complete.")

# ==========================================================
# CLASS: TEMPLATE MASTER
# Handles extracting templates from Video 1 and scanning Video N
# ==========================================================
class TemplateMaster:
    def __init__(self):
        self.saved_templates = [] # List of numpy arrays (images)
        self.template_dims = []   # List of (w, h)

    def collect_templates_from_user(self, frame):
        """Opens GUI for user to draw boxes on the reference frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        print("\n------------------------------------------------")
        print(" INITIAL SETUP (FIRST VIDEO ONLY)")
        print(" 1. Draw boxes around the types of particles you want to track.")
        print(" 2. The script will save these 'Appearances'.")
        print(" 3. It will use these to auto-detect particles in ALL future videos.")
        print(" 4. Close window when done.")
        print("------------------------------------------------")

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(frame_rgb)
        ax.set_title("Define Particle Templates (First Video)")
        
        current_selection = []

        def on_select(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            if (x_max - x_min) < 5 or (y_max - y_min) < 5: return
            
            # Save the crop
            template_img = frame_gray[y_min:y_max, x_min:x_max]
            self.saved_templates.append(template_img)
            self.template_dims.append((x_max-x_min, y_max-y_min))
            
            # Visual feedback
            rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            fig.canvas.draw()
            print(f"  -> Template {len(self.saved_templates)} saved.")

        selector = RectangleSelector(ax, on_select, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='data', interactive=True)
        plt.show()

        if not self.saved_templates:
            raise ValueError("No templates selected! Cannot proceed.")
        
        print(f"-> Saved {len(self.saved_templates)} reference templates for batch processing.")

    def scan_frame(self, frame):
        """Scans a new frame using the saved templates."""
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = frame_gray.shape
        
        detected_points = {} # {id: (cx, cy)}
        obj_id = 1
        
        print(f"  -> Scanning frame using {len(self.saved_templates)} saved templates...")
        
        for idx, template in enumerate(self.saved_templates):
            t_h, t_w = template.shape
            
            # Safety check: if template is larger than video (unlikely but possible if videos differ)
            if t_h > frame_h or t_w > frame_w:
                print(f"    Warning: Template {idx} is larger than current video frame. Skipping.")
                continue

            # Template Matching
            res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= MATCH_THRESHOLD)
            
            new_matches = 0
            for pt in zip(*loc[::-1]):
                cx = pt[0] + t_w // 2
                cy = pt[1] + t_h // 2
                
                # Check overlap
                is_new = True
                min_dist = min(t_w, t_h) / 1.5
                
                for _, (ex, ey) in detected_points.items():
                    dist = np.sqrt((cx - ex)**2 + (cy - ey)**2)
                    if dist < min_dist:
                        is_new = False
                        break
                
                if is_new:
                    detected_points[obj_id] = (cx, cy)
                    obj_id += 1
                    new_matches += 1
            
            # print(f"    -> Template {idx+1} matched {new_matches} times.")

        print(f"  -> Total Unique Matches Found: {len(detected_points)}")
        return detected_points

# ==========================================================
# SAM 3 PROCESSING
# ==========================================================
def get_centroid_and_area(mask):
    if mask is None: return None, 0
    area_pixels = np.count_nonzero(mask)
    rows, cols = np.where(mask)
    if len(rows) == 0: return None, 0
    center_y = int(np.mean(rows))
    center_x = int(np.mean(cols))
    return (center_x, center_y), area_pixels

def overlay_visuals(image, frame_data, colors):
    overlay = image.copy()
    for obj_id, data in frame_data.items():
        cx, cy = data['centroid']
        color = colors.get(obj_id, (0, 255, 0))
        cv2.circle(overlay, (cx, cy), 3, color, -1)
        cv2.putText(overlay, str(obj_id), (cx+4, cy-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    return overlay

def process_chunk(predictor, chunk_path, save_path, fps, prompts_dict, chunk_idx, total_frames_per_chunk, csv_path):
    chunk_vis_data = {}
    last_frame_masks = {}
    all_ids = list(prompts_dict.keys())
    
    csv_file = open(csv_path, mode='a', newline='')
    writer = csv.writer(csv_file)
    
    try:
        # BATCH LOOP
        for i in range(0, len(all_ids), BATCH_SIZE):
            batch = all_ids[i : i + BATCH_SIZE]
            
            force_gpu_cleanup()
            inference_state = predictor.init_state(video_path=chunk_path)
            
            for oid in batch:
                nx, ny = prompts_dict[oid]
                pt = torch.tensor([[nx, ny]], dtype=torch.float32)
                lbl = torch.tensor([1], dtype=torch.int32)
                predictor.add_new_points(inference_state, 0, oid, pt, lbl)
            
            for f_idx, obj_ids, _, video_res_masks, _ in predictor.propagate_in_video(
                inference_state, start_frame_idx=0, max_frame_num_to_track=MAX_TRACK_FRAMES, reverse=False, propagate_preflight=True
            ):
                if f_idx not in chunk_vis_data: chunk_vis_data[f_idx] = {}
                
                for idx, out_oid in enumerate(obj_ids):
                    mask = (video_res_masks[idx] > 0.0).cpu().numpy().squeeze()
                    center, area_px = get_centroid_and_area(mask)
                    
                    if center:
                        cx, cy = center
                        area_um = area_px * PIXEL_AREA_CONVERSION
                        
                        g_frame = (chunk_idx * total_frames_per_chunk) + f_idx
                        writer.writerow([g_frame, out_oid, cx, cy, area_px, area_um])
                        
                        chunk_vis_data[f_idx][out_oid] = {'centroid': (cx, cy)}
                        last_frame_masks[out_oid] = mask

            predictor.clear_all_points_in_video(inference_state)
            del inference_state
            force_gpu_cleanup()

        csv_file.close()
        
        cap = cv2.VideoCapture(chunk_path)
        h, w = int(cap.get(4)), int(cap.get(3))
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        colors = {oid: get_random_color(oid) for oid in all_ids}
        
        curr = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if curr in chunk_vis_data:
                frame = overlay_visuals(frame, chunk_vis_data[curr], colors)
            out.write(frame)
            curr += 1
        cap.release()
        out.release()
        
        return last_frame_masks

    except Exception as e:
        csv_file.close()
        raise e

# ==========================================================
# ANALYSIS
# ==========================================================
def generate_final_report(video_name, output_dir, csv_path, width, height):
    print(f"  -> Generating Analysis Reports for {video_name}...")
    try:
        df = pd.read_csv(csv_path)
        if df.empty: return

        # 1. Histogram
        stats = df.groupby('Object_ID')['Size_um2'].mean()
        plt.figure(figsize=(10, 6))
        plt.hist(stats, bins=30, color='royalblue', edgecolor='black', alpha=0.7)
        plt.title(f"Particle Size Distribution (Total Count: {len(stats)})")
        plt.xlabel(r"Area ($\mu m^2$)")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.3)
        
        text_str = '\n'.join((
            rf"Count: {len(stats)}",
            rf"Mean: {stats.mean():.2f} $\mu m^2$",
            rf"Median: {stats.median():.2f} $\mu m^2$",
            rf"Std Dev: {stats.std():.2f}"
        ))
        plt.text(0.95, 0.95, text_str, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.savefig(os.path.join(output_dir, f"{video_name}_size_histogram.png"))
        plt.close()

        # 2. Trajectories
        plt.figure(figsize=(10, 6))
        plt.gca().invert_yaxis()
        for oid, group in df.groupby('Object_ID'):
            plt.plot(group['Centroid_X'], group['Centroid_Y'], linewidth=0.5, alpha=0.6)
        plt.title("Particle Trajectories")
        plt.xlabel("X (px)")
        plt.ylabel("Y (px)")
        plt.xlim(0, width)
        plt.ylim(height, 0)
        plt.savefig(os.path.join(output_dir, f"{video_name}_trajectories.png"))
        plt.close()

        # 3. Heatmap
        plt.figure(figsize=(10, 6))
        plt.hist2d(df['Centroid_X'], df['Centroid_Y'], bins=[50, 50], cmap='inferno', range=[[0, width], [0, height]])
        plt.colorbar(label='Frequency')
        plt.title("Particle Presence Heatmap")
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(output_dir, f"{video_name}_heatmap.png"))
        plt.close()

        # 4. Summary Stats
        stats_df = df.groupby('Object_ID').agg({
            'Size_um2': ['mean', 'std'],
            'Global_Frame_ID': 'count'
        })
        stats_df.columns = ['Mean_Size_um2', 'Std_Size_um2', 'Frames_Tracked']
        stats_df.to_csv(os.path.join(output_dir, f"{video_name}_summary_stats.csv"))
        
        print(f"  -> Analysis Saved to {output_dir}")

    except Exception as e:
        print(f"Error generating analysis: {e}")
        traceback.print_exc()

# ==========================================================
# MAIN
# ==========================================================
def main():
    setup_environment()
    print("==================================================")
    print(" SAM 3 AUTOMATED BATCH PIPELINE (ONE-TIME SETUP)")
    print("==================================================")
    
    # 1. Load SAM 3
    print("-> Loading SAM 3 Model...")
    with torch.inference_mode():
        sam3_model = build_sam3_video_model()
        predictor = sam3_model.tracker
        predictor.backbone = sam3_model.detector.backbone
    
    # 2. Find Videos
    video_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.mp4")))
    if not video_files:
        print(f"No .mp4 files found in {INPUT_FOLDER}")
        return
    print(f"-> Found {len(video_files)} videos to process.")

    # 3. INITIAL SETUP (Process Video 1 just to get the crops)
    print(f"\n-> Loading first video for Template Selection: {os.path.basename(video_files[0])}")
    
    # We extract frame 0 from the first video
    cap = cv2.VideoCapture(video_files[0])
    _, frame0 = cap.read()
    cap.release()
    
    # Initialize the Master Class and get user input
    template_master = TemplateMaster()
    try:
        template_master.collect_templates_from_user(frame0)
    except Exception as e:
        print(f"Setup failed: {e}")
        return

    # 4. BATCH PROCESS LOOP
    for v_idx, video_path in enumerate(video_files):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n[{v_idx+1}/{len(video_files)}] Processing: {video_name}")
        
        # Setup Dirs
        video_output_dir = os.path.join(MAIN_OUTPUT_DIR, video_name)
        temp_chunk_dir = os.path.join(video_output_dir, "temp_chunks")
        processed_chunk_dir = os.path.join(video_output_dir, "processed_chunks")
        
        if not os.path.exists(video_output_dir): os.makedirs(video_output_dir)
        if not os.path.exists(processed_chunk_dir): os.makedirs(processed_chunk_dir)
        
        csv_path = os.path.join(video_output_dir, f"{video_name}_tracking_data.csv")
        final_video_path = os.path.join(video_output_dir, f"{video_name}_tracked.mp4")

        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Global_Frame_ID", "Object_ID", "Centroid_X", "Centroid_Y", "Size_Pixels", "Size_um2"])

        # A. Split
        chunk_paths, w, h, fps = split_video(video_path, temp_chunk_dir, CHUNK_DURATION)
        frames_per_chunk = int(fps * CHUNK_DURATION)

        # B. Auto-Detect (Using Saved Templates)
        try:
            # Load first frame of THIS video
            cap = cv2.VideoCapture(chunk_paths[0])
            _, first_frame = cap.read()
            cap.release()
            
            # Scan using the Master templates
            raw_points = template_master.scan_frame(first_frame)
            
            if not raw_points:
                print("    ! No matching particles found in this video. Skipping.")
                shutil.rmtree(temp_chunk_dir)
                shutil.rmtree(processed_chunk_dir)
                continue
                
            # Normalize for SAM
            current_prompts = {oid: (c[0]/w, c[1]/h) for oid, c in raw_points.items()}
            
        except Exception as e:
            print(f"Detection error: {e}")
            traceback.print_exc()
            continue

        # C. Track
        print(f"  -> Tracking {len(current_prompts)} objects...")
        
        for i, chunk_path in enumerate(chunk_paths):
            chunk_filename = os.path.basename(chunk_path)
            processed_path = os.path.join(processed_chunk_dir, f"processed_{chunk_filename}")
            
            try:
                final_masks = process_chunk(
                    predictor, chunk_path, processed_path, fps, 
                    current_prompts, i, frames_per_chunk, csv_path
                )
                
                next_prompts = {}
                for oid in current_prompts.keys():
                    if oid in final_masks:
                        center, _ = get_centroid_and_area(final_masks[oid])
                        if center:
                            next_prompts[oid] = (center[0]/w, center[1]/h)
                
                current_prompts = next_prompts
                print(f"    -> Chunk {i+1}: {len(current_prompts)} survived.")
                
                if not current_prompts:
                    break

            except Exception as e:
                print(f"    -> Error in chunk {i}: {e}")
                traceback.print_exc()
                break
        
        # D. Stitch & Analyze
        stitch_processed_chunks(processed_chunk_dir, final_video_path, fps, w, h)
        generate_final_report(video_name, video_output_dir, csv_path, w, h)
        
        # E. Clean
        print("  -> Cleaning temp files...")
        shutil.rmtree(temp_chunk_dir)
        shutil.rmtree(processed_chunk_dir)
        
        print(f"-> Completed {video_name}")

    print("\n==================================================")
    print(f"Pipeline Complete. Results: {MAIN_OUTPUT_DIR}")
    print("==================================================")

if __name__ == "__main__":
    main()