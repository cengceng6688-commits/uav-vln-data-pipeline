import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

def analyze_video_scores(video_path, output_csv="frame_scores.csv"):
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    print(f"[INFO] Analyzing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0:
        print("[ERROR] Cannot read video FPS.")
        return
        
    ret, frame1 = cap.read()
    if not ret:
        print("[ERROR] Cannot read the first frame.")
        return
        
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    scores = []
    timestamps = []
    frame_indices = []
    
    frame_idx = 1
    
    # Iterate through frames to calculate the difference score
    while True:
        ret, frame2 = cap.read()
        if not ret: break
        
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(next_frame, prvs)
        score = np.mean(diff)
        
        scores.append(score)
        timestamps.append(frame_idx / fps)
        frame_indices.append(frame_idx)
        
        prvs = next_frame
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Progress: {frame_idx}/{total_frames} frames...")

    cap.release()
    print("[INFO] Video reading complete, generating report...")

    # 1. Write data to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'Time(s)', 'Score'])
        for f, t, s in zip(frame_indices, timestamps, scores):
            writer.writerow([f, round(t, 3), round(s, 3)])
    print(f"[SUCCESS] Data saved to: {output_csv}")

    # 2. Plot line chart
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, scores, label='Frame Difference Score', color='b', linewidth=1)
    
    # Draw reference lines for intuition
    plt.axhline(y=5, color='orange', linestyle='--', label='Default Motion Threshold (5)')
    plt.axhline(y=55, color='r', linestyle='--', label='Default Shake Threshold (55)')
    
    plt.title(f"Frame Difference Analysis\n{os.path.basename(video_path)}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Score (Mean Absolute Difference)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Replace with a typical test video path 
    # (preferably containing still moments, start of motion/speech, and shaking)
    TEST_VIDEO_PATH = "/Users/zzj/Desktop/vln/mp4_1/16c2c9d480b5decf531b0431ec6d4e9f.mp4" 
    
    analyze_video_scores(TEST_VIDEO_PATH, output_csv="video_analysis_report.csv")