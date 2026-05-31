from pathlib import Path
from uuid import uuid4
import cv2

def extract_frames(input_directory, output_directory, size=(224, 224), VIDEO_EXTENSIONS={".mp4", ".avi"}):
    
    # directory configs
    input_directory = Path(input_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)
    
    # loop through each video in directory
    for video_path in input_directory.rglob("*"):
        
        # skip non-video files
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        
        # capture/load video from video path
        video = cv2.VideoCapture(str(video_path))
        
        # get video fps
        fps = video.get(cv2.CAP_PROP_FPS)
        
        # fallback fps
        if fps is None or fps <= 0: fps = 25
        
        interval = int(fps)
        
        frame_no = 0
        
        # loop through the video
        while True:
            
            # read next frame
            ret, frame = video.read()
            # ret -> True/False of video load
            # frame -> Actual Image
            
            # stop after video ends (ret==False)
            if not ret: break            
            
            # resize frame
            frame = cv2.resize(frame, size)
            
            # frame selection logic
            # 1 frame per second
            if frame_no%interval == 0:
                
                # unique name to avoid confilcts
                filename = f"{video_path.stem}_{frame_no}_{uuid4().hex}.jpg"
                
                # write to output directory
                cv2.imwrite(
                    str(output_directory/filename),
                    frame
                )                
            
            frame_no+=1
            
        # release video and frees memory
        video.release()
        print(f"Done {video_path.name}")
    