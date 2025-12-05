import os
import cv2
import torch
import numpy as np
import shutil
import csv
import gc
import time
import sys
from PIL import Image

# Memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class SAM3ImagePipeline:
    def __init__(self, output_dir, config, log_signal=None):
        self.output_dir = output_dir
        self.config = config
        self.log_signal = log_signal
        
        self.model = None
        self.processor = None
        self.inference_state = None
        
        self.current_fps = 0.0
        self.active_objects = 0
        self.stop_flag = False
        
        self.manual_prompts = [] # List of [x, y, w, h] in pixels

    def log(self, msg):
        if self.log_signal: self.log_signal.emit(msg)
        else: print(msg)

    def force_gpu_cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def load_model(self):
        self.log("Importing SAM 3 Image modules...")
        sam_path = self.config.get('sam_path', '')
        
        # Locate BPE file (Required for Text Prompts)
        bpe_path = os.path.join(sam_path, "assets", "bpe_simple_vocab_16e6.txt.gz")
        if not os.path.exists(bpe_path):
             # Try one level up just in case
             bpe_path = os.path.join(sam_path, "..", "assets", "bpe_simple_vocab_16e6.txt.gz")
        
        if not os.path.exists(bpe_path):
            self.log(f"WARNING: BPE Vocab file not found at {bpe_path}. Text prompts might fail.")

        try:
            import sam3
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            self.log("Building SAM 3 Image Model...")
            
            # Threshold is set during Processor init, NOT set_text_prompt
            conf_thresh = self.config.get('match_threshold', 0.50)
            
            with torch.inference_mode():
                self.model = build_sam3_image_model(bpe_path=bpe_path) 
                self.processor = Sam3Processor(self.model, confidence_threshold=conf_thresh)
                
            self.log("Image Model Loaded Successfully.")
        except Exception as e:
            raise Exception(f"Model Load Failed: {e}")

    def prepare_manual_prompts(self, templates_dict):
        """
        Converts UI QRects to list of [x,y,w,h] (Pixels, Top-Left).
        """
        self.manual_prompts = []
        if not templates_dict: return
        
        for f_idx, rects in templates_dict.items():
            for r in rects:
                # x, y, w, h
                self.manual_prompts.append([r.x(), r.y(), r.width(), r.height()])
        
        self.log(f"Loaded {len(self.manual_prompts)} manual box prompts.")

    def process_video(self, video_path, progress_callback):
        name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(self.output_dir, name)
        if not os.path.exists(out_dir): os.makedirs(out_dir)

        csv_path = os.path.join(out_dir, f"{name}_data.csv")
        vid_path = os.path.join(out_dir, f"{name}_segmented.mp4")

        # Header compatible with YOLO Creator
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(["Global_Frame_ID", "Object_ID", "Centroid_X", "Centroid_Y", "Size_Pixels", "Size_um2"])

        text_prompt = self.config.get('text_prompt', "").strip()
        pixel_scale_sq = self.config.get('pixel_scale_um', 0.324) ** 2
        save_video = self.config.get('save_video', True)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        writer = None
        if save_video:
            writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_vid, h_vid))

        frame_idx = 0
        start_time = time.time()

        try:
            while True:
                if self.stop_flag: break
                ret, frame = cap.read()
                if not ret: break
                
                # 1. Prepare Image (BGR -> RGB -> PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                with torch.inference_mode():
                    # 2. Set Image
                    self.inference_state = self.processor.set_image(pil_img)
                    self.processor.reset_all_prompts(self.inference_state)
                    
                    prompts_added = False

                    # 3. Text Prompt
                    if text_prompt:
                        self.inference_state = self.processor.set_text_prompt(
                            state=self.inference_state, 
                            prompt=text_prompt
                        )
                        prompts_added = True

                    # 4. Box Prompts (MATH FIX)
                    # SAM3 Image expects Normalized Center Coordinates [cx, cy, w, h]
                    if self.manual_prompts:
                        for box in self.manual_prompts:
                            # Input: [x_top_left, y_top_left, width, height] (Pixels)
                            x, y, w_box, h_box = box
                            
                            # Convert to Center (Pixels)
                            cx = x + (w_box / 2.0)
                            cy = y + (h_box / 2.0)
                            
                            # Normalize (0.0 to 1.0)
                            norm_box = [
                                cx / w_vid,
                                cy / h_vid,
                                w_box / w_vid,
                                h_box / h_vid
                            ]
                            
                            self.inference_state = self.processor.add_geometric_prompt(
                                state=self.inference_state, 
                                box=norm_box, 
                                label=True
                            )
                        prompts_added = True

                    # 5. Retrieve Results
                    if not prompts_added:
                        if writer: writer.write(frame)
                        frame_idx += 1
                        continue

                    masks_tensor = self.inference_state.get("masks", None)
                    
                    csv_buffer = []
                    frame_objects = 0
                    
                    if masks_tensor is not None:
                        # Masks shape: [N, 1, H, W] or [N, H, W]
                        masks_np = masks_tensor.cpu().numpy()
                        if len(masks_np.shape) == 4: masks_np = masks_np[:, 0, :, :]
                        
                        for i, mask in enumerate(masks_np):
                            bin_mask = mask > 0.0
                            
                            # Get Stats
                            center, area_px = self._get_centroid_area(bin_mask)
                            
                            if center:
                                frame_objects += 1
                                cx, cy = center
                                area_um = area_px * pixel_scale_sq
                                
                                # Save to CSV
                                csv_buffer.append([frame_idx, i, cx, cy, area_px, area_um])
                                
                                # Draw Visuals
                                if writer:
                                    color = (0, 255, 0)
                                    contours, _ = cv2.findContours(bin_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    cv2.drawContours(frame, contours, -1, color, 2)
                                    cv2.circle(frame, center, 2, (0,0,255), -1)

                    self.active_objects = frame_objects
                    
                    if csv_buffer:
                        with open(csv_path, 'a', newline='') as f:
                            csv.writer(f).writerows(csv_buffer)

                if writer: writer.write(frame)
                
                # Updates
                if frame_idx % 5 == 0:
                    pct = int((frame_idx / total_frames) * 100)
                    progress_callback(pct, 100, f"Processing Frame {frame_idx}/{total_frames}")
                    dur = time.time() - start_time
                    if dur > 0: self.current_fps = frame_idx / dur

                if frame_idx % 50 == 0: self.force_gpu_cleanup()
                frame_idx += 1

            progress_callback(100, 100, "Done")
            
        except Exception as e:
            self.log(f"Image Pipeline Error: {e}")
            raise e
        finally:
            cap.release()
            if writer: writer.release()
            self.force_gpu_cleanup()

    def _get_centroid_area(self, mask):
        if mask is None: return None, 0
        rows, cols = np.where(mask)
        if len(rows) == 0: return None, 0
        return (int(np.mean(cols)), int(np.mean(rows))), len(rows)

    def cleanup(self):
        if self.model: del self.model
        if self.processor: del self.processor
        self.force_gpu_cleanup()