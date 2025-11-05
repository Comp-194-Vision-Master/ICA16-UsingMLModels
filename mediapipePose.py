"""
File: mediapipePose.py
Date: Fall 2025

This program provides a demo showing how to use Mediapipe's body pose detection model, and visualize the results.
"""

import cv2
import numpy as np

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def runPoseDetector(source=0):
    """Sets up the pose landmark model, and runs it on a video feed, visualizing the results"""

    # Set up model
    modelPath = "MediapipeModels/Pose landmark detection/pose_landmarker_full.task"
    base_options = python.BaseOptions(model_asset_path=modelPath)
    options = vision.PoseLandmarkerOptions(base_options=base_options,
                                           output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    # Set up camera
    cap = cv2.VideoCapture(source)

    i = 0
    while True:
        gotIm, frame = cap.read()
        if not gotIm:
            break

        # Convert the frame to be a Mediapipe image format
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Run the pose detector on the image
        detect_result = detector.detect(mp_image)

        # TODO: Uncomment this call to run the function that checks if that hands are above the head or not
        # findHandsUp(detect_result)

        # Visualize the pose skeleton on the frame
        annot_image = visualizeResults(mp_image.numpy_view(), detect_result)
        vis_image = cv2.cvtColor(annot_image, cv2.COLOR_RGB2BGR)

        # If image segementation was done, display the segmentation masks
        segMasks = detect_result.segmentation_masks
        if segMasks is not None and len(segMasks) > 0:
            segIm = segMasks[0].numpy_view()
            cv2.imshow("SegMask", segIm)
            segWrit = 255 * segIm
            cv2.imshow("segWrit", segWrit)
            if i % 30 == 0:
                cv2.imwrite("poseSegm" + str(i) + ".png", segWrit)
        cv2.imshow("Detected", vis_image)

        x = cv2.waitKey(10)
        if x > 0:
            if chr(x) == 'q':
                break
        if i % 30 == 0:
            cv2.imwrite("poseSkel" + str(i) + ".png", vis_image)
        i += 1
    cap.release()


def visualizeResults(rgb_image, detection_result):
    """
    Draws the pose skeleton on a copy of the input image, based on the data in detection_result
    :param rgb_image: an image in RGB format
    :param detection_result: The results of the pose landmark detector
    :return: a copy of the input image with the pose drawn on it
    """
    annotated_image = np.copy(rgb_image)
    pose_landmarks_list = detection_result.pose_landmarks

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image
    plt.show()


def findHandsUp(detect_result):
    """Takes in the pose landmark information, and for each body detected, determines if the hands are
    above the head or not. Prints a message."""
    # TODO: Look at the hand and head positions and determine whether hands are above head or not
    pass


if __name__ == "__main__":
    runPoseDetector("../../SampleVideos/womanBeach.mp4")
