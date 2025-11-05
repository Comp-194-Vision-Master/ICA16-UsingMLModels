"""
File: mediapipeObjects.py
Date: Fall 2025

This program provides a demo showing how to use Mediapipe's object detection model
"""

import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def runObjects():
    """Sets up one of the object detection models and runs it on a video feed."""

    # Set up model
    modelPath = "MediapipeModels/Object Detection (Coco)/efficientdet_lite2.tflite"
    base_options = python.BaseOptions(model_asset_path=modelPath)
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                           score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)

    # Set up camera
    cap = cv2.VideoCapture(0)

    while True:
        gotIm, frame = cap.read()
        if not gotIm:
            break

        # Convert the video frame to a Mediapipe image format
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Run the object detector model on the image
        detect_result = detector.detect(mp_image)

        # Display the results
        annot_image = visualizeResults(mp_image.numpy_view(), detect_result)
        vis_image = cv2.cvtColor(annot_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Detected", vis_image)

        x = cv2.waitKey(10)
        if x > 0:
            if chr(x) == 'q':
                break
    cap.release()


def visualizeResults(image, detection_result):
    """Draws bounding boxes on the input image and return it.
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualized.
    Returns:
      Image with bounding boxes.
    """
    image = image.copy()
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    return image


if __name__ == "__main__":
    runObjects()
