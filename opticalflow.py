import numpy as np
import cv2
import os
import glob

# Initialise the video we want to work on
video_path = 'Opticaltest.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    # try to find any mp4 in the current directory as a fallback
    dir = glob.glob('*.mp4') + glob.glob('*.MP4')
    if dir:
        print(f"Warning: couldn't open {video_path!r}, using {dir[0]!r} instead")
        cap = cv2.VideoCapture(dir[0])
    else:
        raise FileNotFoundError(f"Could not open video file {video_path!r} and no .mp4 files found in {os.getcwd()}")
feature_params = dict(
    maxCorners=50,
    qualityLevel=0.03,
    minDistance=20,
    blockSize=20,
)
# Set parameters for Lucas-Kanade method (pyramidal)
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)
# Create some random colors for feature points
color = np.random.randint(0, 255, (100, 3))
# Read the first frame and find corners in it
ret, old_frame = cap.read()
if not ret or old_frame is None:
    cap.release()
    raise RuntimeError(f"Failed to read the first frame from video {video_path!r}. Check the file and try again.")
# Get video properties for output
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Create VideoWriter to save the tracked video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Opticaltracked.mp4', fourcc, fps, (width, height))
# Convert the first frame to grayscale and preprocess
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# Use the Shi-Tomasi method to detect good features to track
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
# Loop through frames in the video
while cap.isOpened():
    # Read the current frame
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    # Convert the frame to grayscale and preprocess
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate optical flow using the Lucas-Kanade method
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if p1 is None or st is None:
        break
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    # Draw the tracks on the frame
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a_i, b_i, c_i, d_i = int(a), int(b), int(c), int(d)
        mask = cv2.line(mask, (a_i, b_i), (c_i, d_i), color[i % len(color)].tolist(), 2)
        frame = cv2.circle(frame, (a_i, b_i), 4, color[i % len(color)].tolist(), -1)
    # Combine the frame and mask to visualize the tracks
    img = cv2.add(frame, mask)
    # Write the frame to the output video
    out.write(img)
    # Display the frame with tracks
    cv2.imshow('frame', img)
    # Check for the 'Esc' key to exit the loop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Update the previous frame and points for the next iteration
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
# Release the video capture and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()