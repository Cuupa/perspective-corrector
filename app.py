from sys import argv

import cv2
import numpy as np
from flask import Flask, request, Response

app = Flask(__name__)


def pre_processing(img):
    """
    Does some pre processing to determine the document borders
    Does greyscaling, gaussian blur, cannary, dialation and erodation
    :param img: the image
    :return: the preprocessed image
    """
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_grey, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 150, 150)
    kernel = np.ones((5, 5))
    img_dialation = cv2.dilate(img_canny, kernel, iterations=2)
    return cv2.erode(img_dialation, kernel, iterations=1)


def get_contours(img):
    max_area = 0
    biggest_contour = np.array([])
    contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            contour_perimeter = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, 0.02 * contour_perimeter, True)
            if len(approximation) == 4 and area > max_area:
                biggest_contour = approximation
                max_area = area
    return biggest_contour


def reorder(points):
    reshaped_points = points.reshape((4, 2))
    reordered_points = np.zeros((4, 1, 2), np.int32)
    added = reshaped_points.sum(1)
    reordered_points[0] = reshaped_points[np.argmin(added)]
    reordered_points[3] = reshaped_points[np.argmax(added)]
    diff = np.diff(reshaped_points, 1)
    reordered_points[1] = reshaped_points[np.argmin(diff)]
    reordered_points[2] = reshaped_points[np.argmax(diff)]
    return reordered_points


def get_warp(img, biggest_contour):
    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]
    point1 = np.float32(reorder(biggest_contour))
    point2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    perspective = cv2.getPerspectiveTransform(point1, point2)
    return cv2.warpPerspective(img, perspective, (width, height))


def get_image(upload):
    data = upload.read()
    np_array = np.frombuffer(data, np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)


@app.route("/api/image/transform", methods=['POST'])
def transform():
    files = request.files
    if not len(files):
        return Response("MISSING FILE, CHECK IF CONTENT TYPE IS MULTIPART FORM", 400)
    upload = files['key'] if 'key' in files else list(files.values())[0]
    if not upload:
        return Response("MISSING FILE", 400)

    img = get_image(upload)
    if img is None:
        return Response("INVALID IMAGE FILE", 400)

    preprocessed_img = pre_processing(img)
    biggest_contour_result = get_contours(preprocessed_img)
    final_image = get_warp(img, biggest_contour_result)
    return Response(final_image)


@app.route("/status")
def status():
    return ""


def get_port():
    for arg in range(1, len(argv)):
        if arg == "-port":
            return argv[arg + 1]
    return 8080


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=get_port())
