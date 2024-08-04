# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 12:39:39 2024

@author: homap
"""

# Lab Task 1: Setup and Basic Extraction

# Objective:

# Install the necessary tools and libraries, and extract frame information from a video.

# Steps:

# 1. Install ffmpeg and ffmpeg-python:Install the ffmpeg tool and the ffmpeg-python library.

#2. Extract Frame Information:Extract frame information from a sample video.

import ffmpeg
video_path = 'X:/7th sem/IVA/LAB/sample.mp4'

# PROBE
probe = ffmpeg.probe(video_path)

# VIDEO STREAM INFO
video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']

# FRAME INFORMATION
for stream in video_streams:
    print(f"Stream index: {stream['index']}")
    print(f"Codec: {stream['codec_name']}")
    print(f"Width: {stream['width']}")
    print(f"Height: {stream['height']}")
    print(f"Frame rate: {stream['r_frame_rate']}")
    print(f"Number of frames: {stream['nb_frames']}")
    print(f"Duration (seconds): {stream['duration']}")
    print()


#Lab Task 2: Frame Type Analysis

#Objective:
# Analyze the extracted frame information to understand the distribution of I, P, and B frames in a video.

#Steps:

# 1. Modify the Script:

#Count the number of I, P, and B frames.
import ffmpeg
import subprocess
import json

def get_frame_counts(video):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'frame=pict_type', '-of', 'json', video
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frames_info = json.loads(result.stdout)

    # Count frames by type
    frame_counts = {'I': 0, 'P': 0, 'B': 0}
    for frame in frames_info['frames']:
        frame_type = frame['pict_type']
        if frame_type in frame_counts:
            frame_counts[frame_type] += 1

    return frame_counts

input_video = 'X:/7th sem/IVA/LAB/sample.mp4'
frame_counts = get_frame_counts(input_video)
print(frame_counts)

#Calculate the percentage of each frame type in the video.

total_frames = sum(frame_counts.values())
percentages = {frame_type: (count / total_frames) * 100 for frame_type, count in frame_counts.items()}
print(percentages)

#2. Analyze Frame Distribution:

#Plot the distribution of frame types using a library like matplotlib.

import matplotlib.pyplot as plt

#Plot a pie chart or bar graph showing the distribution of frame types using matplotlib.

#BAR CHART

frame_types = list(frame_counts.keys())
counts = list(frame_counts.values())

plt.figure(figsize=(10, 5))
bars = plt.bar(frame_types, counts, color=['red', 'green', 'blue'])
plt.xlabel('Frame Types')
plt.ylabel('Count')
plt.title('Distribution of Frame Types in Video')
for bar, frame_type in zip(bars, frame_types):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, f'{percentages[frame_type]:.2f}%', ha='center', va='bottom')
plt.show()

#PIE CHART

frame_types = list(frame_counts.keys())
percent_values = list(percentages.values())

plt.figure(figsize=(10, 5))
plt.pie(percent_values, labels=frame_types, colors=['red', 'green', 'blue'], autopct='%1.2f%%', startangle=140)
plt.title('Percentage Distribution of Frame Types in Video')
plt.axis('equal')  
plt.show()

#Lab Task 3: Visualizing Frames

#Objective:

# Extract actual frames from the video and display them using Python.

#Steps:

# 1. Extract Frames:

import ffmpeg
import subprocess
import json
import os
    

# Function to get frame counts and types
def get_frame_counts(video):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'frame=pict_type', '-of', 'json', video
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    frames_info = json.loads(result.stdout)
    
    frame_counts = {'I': 0, 'P': 0, 'B': 0}
    for frame in frames_info['frames']:
        frame_type = frame['pict_type']
        if frame_type in frame_counts:
            frame_counts[frame_type] += 1

    return frame_counts, frames_info['frames']

# Extracted frames into folders
def extract_and_save_frames(video_path, frames_info, output_dirs):
    for index, frame in enumerate(frames_info):
        frame_type = frame['pict_type']
        output_dir = output_dirs.get(frame_type)
        if output_dir:
            output_path = os.path.join(output_dir, f"frame_{index:05d}.jpg")
            (
                ffmpeg
                .input(video_path, ss=index / float(stream['r_frame_rate'].split('/')[0]))
                .output(output_path, vframes=1)
                .run()
            )
            print(f"Saved {frame_type} frame {index} to {output_path}")

# Define paths and create directories
video_path = 'X:/7th sem/IVA/LAB/sample.mp4'
output_dirs = {
    'I': 'X:/7th sem/IVA/LAB/I_frames',
    'P': 'X:/7th sem/IVA/LAB/P_frames',
    'B': 'X:/7th sem/IVA/LAB/B_frames'  
}

frame_counts, frames_info = get_frame_counts(video_path)
print("Frame Counts:", frame_counts)

# EXTRACTION AND SAVING
extract_and_save_frames(video_path, frames_info, output_dirs)

# Save these frames as image files.

# 2. Display Frames:

#Use a library like PIL (Pillow) or opencv-python to display the extracted frames.

#Tasks:

# 1. Save I, P, and B frames as separate image files using ffmpeg. - DONE

# 2. Use PIL or opencv-python to load and display these frames in a Python script.

import cv2
import numpy as np

# Function to display frames
def display_frames(dir_path, frame_type, max_frames=5):
    print(f"Displaying up to {max_frames} {frame_type} frames in one window...")
    frame_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.jpg')]
    displayed_count = 0
    images = []
    
    # Load up to max_frames images
    for frame_file in sorted(frame_files):
        if displayed_count >= max_frames:
            break
        image = cv2.imread(frame_file)
        images.append(image)
        displayed_count += 1
    
    # Combine images into a single window
    if len(images) == 0:
        print(f"No frames found in {dir_path}.")
        return
    
    # Resize images to be the same height
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    max_height = max(heights)
    total_width = sum(widths)
    
    resized_images = [cv2.resize(img, (widths[i], max_height)) for i, img in enumerate(images)]
    
    # Concatenate images horizontally
    combined_image = np.hstack(resized_images)
    
    # Display the combined image
    cv2.imshow(frame_type, combined_image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

# Display up to 5 I-Frames
display_frames(output_dirs['I'], 'I-Frames', max_frames=5)

# Display up to 5 P-Frames
display_frames(output_dirs['P'], 'P-Frames', max_frames=5)

# Display up to 5 B-Frames
display_frames(output_dirs['B'], 'B-Frames', max_frames=5)

# 3. Compare the visual quality of I, P, and B frames.

import cv2
import numpy as np

# Function to calculate PSNR
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Function to load images from a directory
def load_images_from_directory(directory, max_frames=5):
    frame_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg')]
    images = []
    for frame_file in sorted(frame_files)[:max_frames]:
        image = cv2.imread(frame_file)
        if image is not None:
            images.append(image)
    return images

# Load I-frames as reference
i_frames = load_images_from_directory(output_dirs['I'])

# Load P and B frames
p_frames = load_images_from_directory(output_dirs['P'])
b_frames = load_images_from_directory(output_dirs['B'])

# Calculate PSNR values
psnr_values = {'P': [], 'B': []}
for p_frame in p_frames:
    psnr = calculate_psnr(i_frames[0], p_frame)
    psnr_values['P'].append(psnr)

for b_frame in b_frames:
    psnr = calculate_psnr(i_frames[0], b_frame)
    psnr_values['B'].append(psnr)

print("PSNR Values for P-Frames:", psnr_values['P'])
print("PSNR Values for B-Frames:", psnr_values['B'])

# Display PSNR values in a bar chart
import matplotlib.pyplot as plt

frame_types = ['P', 'B']
psnr_means = [np.mean(psnr_values[frame_type]) for frame_type in frame_types]

plt.figure(figsize=(10, 5))
bars = plt.bar(frame_types, psnr_means, color=['green', 'blue'])
plt.xlabel('Frame Types')
plt.ylabel('Average PSNR')
plt.title('Average PSNR of P and B Frames Compared to I-Frames')
for bar, frame_type in zip(bars, frame_types):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{yval:.2f}', ha='center', va='bottom')
plt.show()

# Lab Task 4: Frame Compression Analysis

# Objective:

# Analyze the compression efficiency of I, P, and B frames.

# Steps:

# 1. Calculate Frame Sizes:

# Calculate the file sizes of extracted I, P, and B frames.
# Compare the average file sizes of each frame type.

def calculate_frame_sizes(output_dirs):
    frame_sizes = {'I': [], 'P': [], 'B': []}
    
    for frame_type, dir_path in output_dirs.items():
        frame_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.jpg')]
        for frame_file in frame_files:
            size = os.path.getsize(frame_file)
            frame_sizes[frame_type].append(size)
    
    return frame_sizes


frame_sizes = calculate_frame_sizes(output_dirs)
for frame_type, sizes in frame_sizes.items():
    total_size = sum(sizes)
    avg_size = total_size / len(sizes) if sizes else 0
    print(f"Frame Type: {frame_type}")
    print(f"  Total Size: {total_size} bytes")
    print(f"  Average Size: {avg_size:.2f} bytes")

# Compression Efficiency
# Discuss the role of each frame type in video compression.
# Analyze why P and B frames are generally smaller than I frames.


# Lab Task 5: Advanced Frame Extraction

# Objective:

# Extract frames from a video and reconstruct a part of the video using only I frames.

#Steps:

#1. Extract and Save I Frames:

# Extract I frames from the video and save them as separate image files. - DONE ALREADY

#2. Reconstruct Video:

# Use the extracted I frames to reconstruct a portion of the video.
import os 
import ffmpeg

# Define the path to the I-frames directory and the output video path
i_frames_dir = 'X:/7th sem/IVA/LAB/I_frames'
output_video_path = 'X:/7th sem/IVA/LAB/reconstructed_video.mp4'

# Get a list of I-frame files
i_frame_files = sorted([os.path.join(i_frames_dir, f) for f in os.listdir(i_frames_dir) if f.endswith('.jpg')])

# Create a text file listing all I-frame files
with open('file_list.txt', 'w') as f:
    for frame_file in i_frame_files:
        f.write(f"file '{frame_file}'\n")
        f.write("duration 1\n")  # Set each frame to display for 1 second

# Use ffmpeg to create a video with the specified frame rate
ffmpeg.input('file_list.txt', format='concat', safe=0).output(output_video_path, vcodec='libx264', framerate=1).run()

print(f"Reconstructed video saved to {output_video_path}")


# Create a new video using these I frames with a reduced frame rate.

# Define paths
i_frames_dir = 'X:/7th sem/IVA/LAB/I_frames'
output_video_path = 'X:/7th sem/IVA/LAB/reduced_framerate_video.mp4'
file_list_path = 'file_list.txt'

# Get a list of I-frame files
i_frame_files = sorted([os.path.join(i_frames_dir, f) for f in os.listdir(i_frames_dir) if f.endswith('.jpg')])

# Create a text file listing all I-frame files
with open('file_list.txt', 'w') as f:
    for frame_file in i_frame_files:
        f.write(f"file '{frame_file}'\n")
        f.write("duration 2\n")  # Set each frame to display for 1 second

# Use ffmpeg to create a video with the specified frame rate
ffmpeg.input('file_list.txt', format='concat', safe=0).output(output_video_path, vcodec='libx264', framerate=1).run()

print(f"Reduced frame rate video saved to {output_video_path}")


















    
