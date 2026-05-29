import whisper
import cv2
import numpy as np
import json
import os
import ssl
import subprocess
import threading
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

ssl._create_default_https_context = ssl._create_unverified_context

whisper_lock = threading.Lock()

def detect_video_status(video_path, motion_offset, min_motion_frames, shake_threshold, min_shake_frames, baseline_seconds=3.0):
   
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): 
        cap.release()
        return "error", 0.0
    
    ret, frame1 = cap.read()
    if not ret: 
        cap.release()
        return "error", 0.0
        
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    start_frame = -1
    motion_streak = 0
    shake_streak = 0
    frame_idx = 1
    
    status = "no_motion"
    
    baseline_scores = []
    dynamic_motion_threshold = None
    baseline_frame_count = int(fps * baseline_seconds)

    while True:
        ret, frame2 = cap.read()
        if not ret: break
        
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(next_frame, prvs)
        score = np.mean(diff)
        
        #计算动态基准线
        if frame_idx <= baseline_frame_count:
            baseline_scores.append(score)
            if frame_idx == baseline_frame_count:       
                baseline_base = np.percentile(baseline_scores, 25)
                dynamic_motion_threshold = baseline_base + motion_offset
                print(f"      [DEBUG] 基准分(25th): {baseline_base:.2f}, 动态运动阈值设定为: {dynamic_motion_threshold:.2f}")
        else:
            # 检查严重晃动
            if score > shake_threshold:
                shake_streak += 1
                if shake_streak >= min_shake_frames:
                    cap.release()
                    return "shaking", 0.0
            else:
                shake_streak = 0
                
            #检查运动开始
            if start_frame == -1 and dynamic_motion_threshold is not None:
                if score > dynamic_motion_threshold:
                    motion_streak += 1
                    if motion_streak >= min_motion_frames:
                        start_frame = frame_idx - min_motion_frames + 1
                        status = "success"
                else:
                    motion_streak = 0
                
        prvs = next_frame
        frame_idx += 1
    
    cap.release()
    
    if status == "success":
        return "success", start_frame / fps
    else:
        return "no_motion", 0.0
    
def process_single_video(video_filename, input_dir, success_dir, failure_dir, model, params):
    """
    Process single video logic, intended for multithreading.
    """
    abs_video_path = os.path.abspath(os.path.join(input_dir, video_filename))
    base_name = os.path.splitext(video_filename)[0]
    
    print(f"[INFO] Start detecting: {video_filename} ...")
    
    try:
        # 画面视觉检测
        status, visual_start_time = detect_video_status(
            abs_video_path,
            motion_offset=params['motion_offset'],
            min_motion_frames=params['min_motion_frames'],
            shake_threshold=params['shake_threshold'],
            min_shake_frames=params['min_shake_frames'],
            baseline_seconds=params.get('baseline_seconds', 3.0) 
        )

        if status != "success":
            print(f"[WARNING] {video_filename} detected as invalid ({status}), moving to failure folder.")
            failed_video_path = os.path.join(failure_dir, video_filename)
            shutil.copy2(abs_video_path, failed_video_path)
            
            fail_data = {
                "original_video": video_filename,
                "status": "failed",
                "reason": status
            }
            with open(os.path.join(failure_dir, f"{base_name}.json"), "w", encoding="utf-8") as f:
                json.dump(fail_data, f, ensure_ascii=False, indent=4)
            return fail_data

        print(f"[INFO] {video_filename} visual check passed (Visual Start: {visual_start_time:.2f}s), extracting audio & transcribing...")
        
        # 提取音频并使用 Whisper 分析时间戳
        temp_audio_path = os.path.abspath(os.path.join(success_dir, f"temp_{base_name}.wav"))
        subprocess.run([
            "ffmpeg", "-nostdin", "-i", abs_video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-map", "0:a:0?", temp_audio_path, "-y"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        full_text = ""
        speech_end_time = 0.0
        
        if os.path.exists(temp_audio_path):
            with whisper_lock:
                result = model.transcribe(
                    temp_audio_path,
                    language="zh"
                )
                segments = result.get("segments", [])
                
                if segments:
                    speech_end_time = segments[0]["end"]
                    for i in range(1, len(segments)):
                        if segments[i]["start"] - segments[i-1]["end"] > 2.5:
                            break
                        speech_end_time = segments[i]["end"]
                    
                    speech_end_time += 0.3 
                
                full_text = "".join([seg["text"] for seg in segments]).strip()
            os.remove(temp_audio_path)
        
        # 综合画面与声音，进行最终裁切
        # 取画面开始运动和语音结束两者中更晚的一个时间点
        final_start_time = max(visual_start_time, speech_end_time)
        print(f"      [DEBUG] 画面点: {visual_start_time:.2f}s | 说话结束: {speech_end_time:.2f}s -> 最终裁剪点: {final_start_time:.2f}s")
        
        output_clip_name = f"{base_name}_clip.mp4"
        output_clip_path = os.path.abspath(os.path.join(success_dir, output_clip_name))
        
        subprocess.run([
            "ffmpeg", "-nostdin", "-i", abs_video_path,
            "-ss", str(final_start_time), 
            "-c:v", "libx264", "-c:a", "aac",
            "-map", "0:v:0?", "-map", "0:a:0?", 
            output_clip_path, "-y"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        #记录数据
        video_data = {
            "original_video": video_filename,
            "status": "success",
            "clip_video_name": output_clip_name,
            "full_transcribed_text": full_text,
            "visual_detected_time": round(visual_start_time, 2),
            "speech_ended_time": round(speech_end_time, 2),
            "trimmed_start_time": round(final_start_time, 2)
        }
        
        with open(os.path.join(success_dir, f"{base_name}.json"), "w", encoding="utf-8") as f:
            json.dump(video_data, f, ensure_ascii=False, indent=4)
            
        print(f"[SUCCESS] {video_filename} processing complete!")
        return video_data
        
    except Exception as e:
        print(f"[ERROR] Failed to process {video_filename}: {e}")
        return None

def process_all_videos(input_dir, output_dir, params, max_workers=4):
    if not os.path.exists(input_dir):
        print(f"[ERROR] Input directory not found: '{input_dir}'.")
        return

    success_dir = os.path.join(output_dir, "success")
    failure_dir = os.path.join(output_dir, "failure")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failure_dir, exist_ok=True)
    print(f"[INFO] Created output directory structure: \n - Success: {success_dir}\n - Failure: {failure_dir}")

    valid_exts = ('.mp4',)
    target_videos = [f for f in os.listdir(input_dir) if f.endswith(valid_exts) and not f.endswith('_clip.mp4')]
    target_videos.sort()

    if not target_videos:
        print(f"[WARNING] No valid .mp4 files found in '{input_dir}'.")
        return

    print(f"[INFO] Found {len(target_videos)} videos, preparing multithreading (Workers: {max_workers})...")
    print("[INFO] Loading Whisper AI model (base)...")
    
    model = whisper.load_model("base")
    all_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_video, video, input_dir, success_dir, failure_dir, model, params): video 
            for video in target_videos
        }
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_results.append(result)

    master_json_path = os.path.join(output_dir, "dataset_index.json")
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
        
    print(f"\n=============================================")
    print(f"[INFO] Batch parallel processing complete! Master index generated: {master_json_path}")

if __name__ == "__main__":
    INPUT_FOLDER = "mp4_converted"
    OUTPUT_FOLDER = "final_dataset"
    
    DETECTION_PARAMS = {
        "baseline_seconds": 3.0,        
        "motion_offset": 2.0,           
        "min_motion_frames": 5,         
        "shake_threshold": 55.0,        
        "min_shake_frames": 5           
    }
    
    process_all_videos(
        input_dir=INPUT_FOLDER, 
        output_dir=OUTPUT_FOLDER, 
        params=DETECTION_PARAMS, 
        max_workers=4
    )