import io
import logging
import os
import time
from sys import argv

import PIL.Image as Image
import cv2
import numpy as np
from flask import Flask, request, Response

app = Flask(__name__)


def pre_processing(img, treshold1, treshold2):
    """
    Does some pre processing to determine the document borders
    Does greyscaling, gaussian blur, cannary, dialation and erodation
    :param treshold2:
    :param treshold1:
    :param img: the image
    :return: the preprocessed image
    """
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_grey, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, treshold1, treshold2)
    kernel = np.ones((5, 5))
    img_dialation = cv2.dilate(img_canny, kernel, iterations=2)
    return cv2.erode(img_dialation, kernel, iterations=1)


def get_contours(img):
    max_area = 0
    biggest_contour = np.array([])
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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


def process(img, treshold1, treshold2):
    preprocessed_img = pre_processing(img, treshold1, treshold2)
    biggest_contour_result = get_contours(preprocessed_img)
    return get_warp(img, biggest_contour_result)


def verify_image(img):
    """
    Verifies the image based on the occurring colors. If the number of different colors is too indistinguishable the
    image is likely to be corrupted. This may be subject to finetuning
    :param img:
    :return:
    """
    image = Image.open(io.BytesIO(img))
    image = image.resize((200, 200))
    cluster = {}

    for x in range(image.width):
        for y in range(image.height):
            r, g, b, alpha = image.getpixel((x, y))
            if cluster.get(r) is not None:
                cluster[r] = cluster[r] + 1
            if cluster.get(g) is not None:
                cluster[g] = cluster[g] + 1
            if cluster.get(b) is not None:
                cluster[b] = cluster[b] + 1

            if cluster.get(r) is None:
                cluster[r] = 1
            if cluster.get(g) is None:
                cluster[g] = 1
            if cluster.get(b) is None:
                cluster[b] = 1

    """
    Definitely a single-colored and therefore a corrupted image.
    """
    if len(cluster) == 1:
        logging.error("The image has only one color and is likely to be invalid")
        return False

    """
    If just a few colors are found, check if the color difference is noticeable
    """
    if len(cluster) <= 5:
        keys = cluster.keys()
        if max(keys) - min(keys) <= 5:
            logging.error("The image has only " + str(
                len(cluster)) + " different colors with nearly " +
                          "indistinguishable color differences and is likely to be invalid")
            return False
    return True


@app.route("/api/image/transform", methods=['POST'])
def transform():
    start = time.perf_counter()
    files = request.files
    if not len(files):
        return Response("MISSING FILE, CHECK IF CONTENT TYPE IS MULTIPART FORM", 400)
    upload = files['key'] if 'key' in files else list(files.values())[0]

    if not upload:
        return Response("MISSING FILE", 400)

    image_name, extension = os.path.splitext(upload.filename)
    img = get_image(upload)
    if img is None:
        return Response("INVALID IMAGE FILE", 400)

    threshold1 = 150
    threshold2 = 150

    final_image = process(img, threshold1, threshold2)

    success, img_result = cv2.imencode(extension, final_image)
    success = verify_image(img_result)

    while not success:
        logging.error("Failed to get a valid image with threshold=" + str(threshold1) + " ... retrying")
        final_image = process(img, threshold1, threshold2)
        success, img_result = cv2.imencode(extension, final_image)
        success = verify_image(img_result)
        threshold1 = threshold1 - 20
        threshold2 = threshold2 - 20

        """
        If no valid result seems to be found, return the original image
        """
        if threshold1 < 0 and threshold2 < 0:
            logging.error("Failed to get a valid image. Returning the original image")
            original_image = cv2.imencode(extension, img)
            return Response(original_image.tobytes())

    stop = time.perf_counter()
    logging.error("Image transformation of '" + image_name + "' took " + str(round(stop - start, 2)) + " ms")
    return Response(img_result.tobytes())


@app.route("/api/status")
def status():
    return ""


def get_port():
    for arg in range(1, len(argv)):
        if arg == "-port":
            return argv[arg + 1]
    return 5000


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=get_port())
