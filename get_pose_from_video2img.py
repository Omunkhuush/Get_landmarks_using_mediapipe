from matplotlib import pyplot as plt
import cv2
from typing import Tuple, Union
import math 
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from os.path import isfile, join
import shutil
from colors import bgr_colors
from concurrent.futures import ThreadPoolExecutor
#from functions import get_pose_from_video_save2image

pose_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
p_options = vision.PoseLandmarkerOptions(
    base_options=pose_options,
    output_segmentation_masks=True)
pose_detector = vision.PoseLandmarker.create_from_options(p_options)

hand_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
h_options = vision.HandLandmarkerOptions(base_options=hand_options,
                                       num_hands=2)
hand_detector = vision.HandLandmarker.create_from_options(h_options)

def get_landmarks(imagesPath,images):
    hand_buffer = []
    pose_buffer = []
    imgNames =[]
    for j in images:
        image = mp.Image.create_from_file(imagesPath+j)
        pose_result = pose_detector.detect(image)
        hand_result = hand_detector.detect(image)
        if len(pose_result.pose_landmarks) == 1:
            if len(hand_result.hand_landmarks)==2:
                hand_buffer.append(hand_result)
                pose_buffer.append(pose_result)
                imgNames.append(j)
    return pose_buffer,hand_buffer, imgNames

def filter(pose_buffer,hand_buffer, imgNames, thresh):
    filtered_pose = [pose_buffer[0]]
    filtered_hand = [hand_buffer[0]]
    filtered_imgNames = [imgNames[0]]
    prev_frame_number = int(filtered_imgNames[-1].split('_')[1].split('.')[0])
    for q in range(1, len(imgNames)):
        frame_number = int(imgNames[q].split('_')[1].split('.')[0])
        if frame_number != prev_frame_number + 1:
            filtered_pose.append(pose_buffer[q])
            filtered_hand.append(hand_buffer[q])
            filtered_imgNames.append(imgNames[q])
        k = int(filtered_imgNames[-1].split('_')[1].split('.')[0])
        if frame_number-k >= 5:
            filtered_pose.append(pose_buffer[q])
            filtered_hand.append(hand_buffer[q])
            filtered_imgNames.append(imgNames[q])
        if (frame_number-prev_frame_number) >= thresh:
                filtered_imgNames =[]
                break
        prev_frame_number = frame_number
    return filtered_pose, filtered_hand, filtered_imgNames

def plot_landmarks(filtered_pose,filtered_hand):
    img_width = 512 
    img_height = 512 
    image_size = (img_width,img_height,3)
    blank_image = np.zeros(image_size, dtype=np.uint8) * 255
      
    pose_coordinates = []
    for k, landmark in enumerate(filtered_pose[0].pose_landmarks[0]):
        x = int(landmark.x * img_width)
        y = int(landmark.y * img_height)
        pose_coordinates.append([x,y])
    splited_pose = []
    eyes = [pose_coordinates[8],pose_coordinates[6],pose_coordinates[5],pose_coordinates[4],pose_coordinates[0],pose_coordinates[1],pose_coordinates[2],pose_coordinates[3],pose_coordinates[7]]
    mouth = [pose_coordinates[9]]+[pose_coordinates[10]]
    chest = [pose_coordinates[11],pose_coordinates[23],pose_coordinates[24],pose_coordinates[12],pose_coordinates[11]]
    splited_pose.append(eyes)
    splited_pose.append(mouth)
    splited_pose.append(chest)
    for points in splited_pose:
        for i in range(len(points) - 1):
            cv2.line(blank_image, tuple(points[i]), tuple(points[i + 1]), (198,198, 199,20), 1)
    for r, pose_result in enumerate(filtered_pose):
        arm = []
        arm_coordinates = []
        for k, landmark in enumerate(pose_result.pose_landmarks[0]):
            x = int(landmark.x * img_width)
            y = int(landmark.y * img_height)
            arm_coordinates.append([x,y])
        left_hand = [arm_coordinates[11],arm_coordinates[13],arm_coordinates[15]]
        right_hand = [arm_coordinates[12],arm_coordinates[14],arm_coordinates[16]]
        arm.append(right_hand)
        arm.append(left_hand)

        hand_result = filtered_hand[r]
        if hand_result.handedness[0][0].display_name != 'Right':
            hand_result.hand_landmarks[0],hand_result.hand_landmarks[1] = hand_result.hand_landmarks[1],hand_result.hand_landmarks[0]

        hand_coordinates = []
        for k, i in enumerate(hand_result.hand_landmarks):
            globals()[f"hand_{k}"] = []
            for landmark in i:
                x = int(landmark.x * img_width)
                y = int(landmark.y * img_height)
                globals()[f"hand_{k}"].append([x,y])
            hand_coordinates.append(globals()[f"hand_{k}"])
        for u, hand in enumerate(hand_coordinates):
            finger1 = hand[:5]
            finger2 = [hand[0]]+hand[5:9]
            finger3 = hand[9:13]
            finger4 = hand[13:17]
            finger5 = [hand[0]]+hand[17:21]
            connectLine = [hand[5],hand[9],hand[13],hand[17]]
            splited_fingers = []
            splited_fingers.append(arm[u])
            splited_fingers.append(finger1)
            splited_fingers.append(finger2)
            splited_fingers.append(finger3)
            splited_fingers.append(finger4)
            splited_fingers.append(finger5)
            splited_fingers.append(connectLine)
            hand = hand+arm[u]
            for points in splited_fingers:
                for i in range(len(points) - 1):
                    cv2.line(blank_image, tuple(points[i]), tuple(points[i + 1]), (198,198, 199,20), 2)
            for point in hand:
                cv2.circle(blank_image, tuple(point), 2, bgr_colors[r], -1)
    return blank_image

def get_pose_from_video_save2image(all_files):
    rootPath = all_files[0]
    rootSavePath= all_files[1]
    folder= all_files[2]
    cls= all_files[3]
    video= all_files[4]
    #for video in files:
    #video = files[0]
    print('File name : ',video)
    imageName = video.split('.')[0]+'.png'
    expected_image_path = os.path.join(rootSavePath, folder, cls, imageName)
    if os.path.exists(expected_image_path):
        print('\t This file already converted !')
        return
    imagesPath = './images/train/'+video.split('.')[0]+'/'
    print('\t image path :',imagesPath)
    if os.path.exists(imagesPath):
        shutil.rmtree(imagesPath)
    os.mkdir(imagesPath)
    videoPath = rootPath + cls+'/'+video
    cap  = cv2.VideoCapture(videoPath)
    k = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            #print("Can't receive frame (stream end?). Exiting . . .")
            break
        cv2.imwrite(imagesPath+'frame_'+str(k)+'.png',frame)
        k+=1
    images = [f for f in os.listdir(imagesPath) if isfile(join(imagesPath, f))]
    images = sorted(images, key=lambda x: int(x.split('_')[1].split('.')[0]))
    print('\t getting landmarks...')
    pose_buffer,hand_buffer, imgNames = get_landmarks(imagesPath, images)
    print('\t number of poses = ',len(pose_buffer),'/',len(images))
    if len(pose_buffer) >= 10:
        thresh = int(len(images)*0.6)
        filtered_pose, filtered_hand, filtered_imgNames = filter(pose_buffer,hand_buffer, imgNames,thresh)
        print('\t threshold = ',thresh)
        print('\t after filter = ',len(filtered_pose),'/',len(pose_buffer))
        if filtered_imgNames == []:
            return
        print('\t ploting...')
        blank_image = plot_landmarks(filtered_pose,filtered_hand)
        savePath = rootSavePath+folder+'/'+cls+'/'
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        cv2.imwrite(savePath+imageName,blank_image)
        print('true deleted')
        shutil.rmtree(imagesPath)
        print('\t'+savePath,imageName)
    else:
        print("\t results are not satisfactory.")
    if os.path.exists(imagesPath):
        print('else deleted')
        shutil.rmtree(imagesPath)

all_files = []
# folders  = ['train','test','val']
# for folder in folders:
folder ='train'
rootPath = '../chalearn_processed_full/color/'+folder+'/'
rootSavePath = './AUTSL_full_diff_color_all_hand_with_pose/'

classes = [f for f in os.listdir(rootPath)]
for cls in classes:
    files = [f for f in os.listdir(rootPath+cls) if isfile(join(rootPath+cls,f))]
    #get_pose_from_video_save2image(rootPath,rootSavePath,folder,cls,video)
    for video in files:
        all_files.append([rootPath,rootSavePath,folder,cls,video])
print(len(all_files))

num_cpus = os.cpu_count()  

with ThreadPoolExecutor(max_workers=num_cpus) as executor:
    # Process videos in parallel
    executor.map(get_pose_from_video_save2image, all_files)