import os
import subprocess
import concurrent.futures

def process_single_video(video_file, input_dir, output_dir):
    """
    Use ffmpeg copy mode for fast conversion, preserving original aspect ratio and orientation.
    """
    input_path = os.path.abspath(os.path.join(input_dir, video_file))
    base_name = os.path.splitext(video_file)[0]
    output_path = os.path.abspath(os.path.join(output_dir, f"{base_name}.mp4"))
    
    print(f"[INFO] Fast converting: {video_file} ...")
    
    try:
        # Core logic:
        # 1. -nostdin: Prevents ffmpeg from hanging while waiting for input.
        # 2. -c:v copy: Copies the video stream directly (no re-encoding) to retain quality and speed.
        # 3. -c:a aac: Converts audio to AAC for standard MP4 compatibility.
        cmd = [
            "ffmpeg", "-nostdin", "-i", input_path,
            "-c:v", "copy", "-c:a", "aac",
            "-map", "0:v:0?", "-map", "0:a:0?", 
            output_path, "-y"
        ]
        
        # Capture logs without spamming the console
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            return True, f"       [SUCCESS] {base_name}.mp4 converted (Original quality preserved)!"
        else:
            # Print the tail of the error message for debugging if it fails
            return False, f"       [ERROR] {video_file} failed. Error: {result.stderr.strip()[-150:]}"
            
    except Exception as e:
        return False, f"       [CRITICAL] {video_file} exception: {e}"

def convert_mov_to_mp4_fast(input_dir='mov_datasets', output_dir='mp4_converted', max_workers=4):
    valid_exts = ('.mov', '.MOV')
    target_videos = [f for f in os.listdir(input_dir) if f.endswith(valid_exts)]
    target_videos.sort()
    
    if not target_videos:
        print(f"[WARNING] No .mov videos found in directory '{input_dir}'.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Created output directory: {output_dir}")

    print(f"[INFO] Found {len(target_videos)} videos. Starting fast lossless conversion (Threads: {max_workers})...")

    success_count = 0
    fail_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_video, video, input_dir, output_dir): video 
            for video in target_videos
        }
        
        for future in concurrent.futures.as_completed(futures):
            is_success, msg = future.result()
            print(msg)
            if is_success:
                success_count += 1
            else:
                fail_count += 1

    print(f"\n[INFO] Batch processing complete! Success: {success_count}, Failed: {fail_count}.")

if __name__ == "__main__":
    convert_mov_to_mp4_fast(input_dir='mov_datasets', output_dir='mp4_converted', max_workers=4)