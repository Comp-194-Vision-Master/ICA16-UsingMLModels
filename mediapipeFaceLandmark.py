"""
File: mediapipeFaceLandmark.py
Date: Fall 2025

This program provides a demo showing how to use Mediapipe's facial landmark model, and to visualize the results.
"""

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as plt


def runFacialLandmarks(source=0):
    # Set up model
    modelPath = "MediapipeModels/face_landmarker_v2_with_blendshapes.task"
    base_options = python.BaseOptions(model_asset_path=modelPath)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    # Set up camera
    cap = cv2.VideoCapture(source)

    while True:
        gotIm, frame = cap.read()
        if not gotIm:
            break

        # Convert image to Mediapipe image format
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Run facial landmark detector
        detect_result = detector.detect(mp_image)

        # TODO: Uncomment this call to run the function that checks whether eyes are open or closed
        # findEyes(detect_result)

        annot_image = visualizeResults(mp_image.numpy_view(), detect_result)
        vis_image = cv2.cvtColor(annot_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Detected", vis_image)

        x = cv2.waitKey(10)
        if x > 0:
            if chr(x) == 'q':
                break
            if chr(x) == 'b' and len(detect_result.face_landmarks) > 0:
                plot_face_blendshapes_bar_graph(detect_result.face_blendshapes[0])
    cap.release()


def visualizeResults(rgb_image, detection_result):
    """
    Draw the face landmark mesh onto a copy of the input RGB image and returns it
    :param rgb_image: an image in RGB format (as a Numpy array)
    :param detection_result: The results of running the face landmarker model
    :return: a copy of rgb_image with face landmark mesh drawn on it
    """
    annotated_image = np.copy(rgb_image)
    face_landmarks_list = detection_result.face_landmarks

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    """
    Creates a plt bar graph to show how much each blendshape is present in a given image
    :param face_blendshapes: output from the blendshapes model
    :return:
    """
    # Extract the face blendshapes category names and scores.
    for face_blsh in face_blendshapes:
        print(face_blsh)
        print(face_blsh.category_name, face_blsh.score)


    face_blsh_names = [face_blsh_category.category_name for face_blsh_category in face_blendshapes]
    face_blsh_scores = [face_blsh_category.score for face_blsh_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blsh_ranks = range(len(face_blsh_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blsh_ranks, face_blsh_scores, label=[str(x) for x in face_blsh_ranks])
    ax.set_yticks(face_blsh_ranks, face_blsh_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blsh_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


def findEyes(detect_result):
    """Takes in the facial landmark results and determines, for each face located, whether the
        eyes are open or closed. Print a message"""
    # TODO: Look at the blendshapes for the eyes and determine if the eyes are open or closed
    pass


if __name__ == "__main__":
    runFacialLandmarks(0)
