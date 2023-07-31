from flask import Flask, request, make_response, send_from_directory

from GHOST import GHOST
import cv2
import numpy as np
import os
import string
import random
import traceback

app = Flask(
    __name__,
    static_url_path='/ghost-face-swap/',
    static_folder='web/dist'
)

ghost = GHOST(
    2,
    'unet',  # 'unet', 'linknet', 'resnet'
    'weights/G_unet_2blocks.pth',
    False,
    224,
    0.15,
    40,
)
ghost.init_models()


@app.route("/ghost-face-swap/")
def hello_world():
    return send_from_directory(app.static_folder, 'index.html')


def image_to_mat(image_file):
    # read image as an numpy array
    image = np.asarray(bytearray(image_file), dtype="uint8")
    # use imdecode function
    return cv2.imdecode(image, cv2.IMREAD_COLOR)

def read_flat_inputs():
    target_storage = request.files.get("target")
    source_face_storage0 = request.files.get("source0")
    source_face_storage1 = request.files.get("source1")
    source_face_storage2 = request.files.get("source2")
    source_face_storage3 = request.files.get("source3")
    source_face_storage4 = request.files.get("source4")
    source_face_storage5 = request.files.get("source5")
    source_face_storage6 = request.files.get("source6")
    source_face_storage7 = request.files.get("source7")

    target_face_storage0 = request.files.get("target0")
    target_face_storage1 = request.files.get("target1")
    target_face_storage2 = request.files.get("target2")
    target_face_storage3 = request.files.get("target3")
    target_face_storage4 = request.files.get("target4")
    target_face_storage5 = request.files.get("target5")
    target_face_storage6 = request.files.get("target6")
    target_face_storage7 = request.files.get("target7")

    source_faces_storages = list(filter(None, [
        source_face_storage0,
        source_face_storage1,
        source_face_storage2,
        source_face_storage3,
        source_face_storage4,
        source_face_storage5,
        source_face_storage6,
        source_face_storage7,
    ]))

    target_faces_storages = list(filter(None, [
        target_face_storage0,
        target_face_storage1,
        target_face_storage2,
        target_face_storage3,
        target_face_storage4,
        target_face_storage5,
        target_face_storage6,
        target_face_storage7,
    ]))

    source_faces = list(map(lambda source_faces_storage: source_faces_storage.read(), source_faces_storages))
    target_faces = list(map(lambda target_faces_storage: target_faces_storage.read(), target_faces_storages))

    return target_storage.read(), source_faces, target_faces


def read_inputs():
    target_storage = request.files.get("target")
    source_faces_storages = request.files.getlist("sources[]")
    target_faces_storages = request.files.getlist("targetFaces[]")

    source_faces = []
    for storage in source_faces_storages:
        source_faces.append(storage.read())
    target_faces = []
    for storage in target_faces_storages:
        target_faces.append(storage.read())

    return target_storage.read(), source_faces, target_faces

def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def get_enlarged_boundaries(width_image, height_image, x, y, width_face, height_face, by_percents = 50):
    x_extra = int(width_face / 100 * by_percents)
    y_extra = int(height_face / 100 * by_percents)

    x1 = x - x_extra
    if x1 < 0:
        x1 = 0

    x2 = x + width_face + x_extra
    if x2 > width_image:
        x2 = width_image

    y1 = y - y_extra
    if y1 < 0:
        y1 = 0

    y2 = y + height_face + y_extra
    if y2 > height_image:
        y2 = height_image

    return x1, x2, y1, y2

def extract_faces(img, mode = None):
    img_mat = image_to_mat(img)
    # Convert into grayscale
    gray = cv2.cvtColor(img_mat, cv2.COLOR_BGR2GRAY)
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    face_images = []
    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
        try:
            x1, x2, y1, y2 = get_enlarged_boundaries(img_mat.shape[1], img_mat.shape[0], x, y, w, h)
            face_mat = img_mat[y1:y2, x1:x2]
            result_array = cv2.imencode('.jpg', face_mat)[1]
            result_encoded = np.array(result_array)
            face_images.append(result_encoded)
            # cv2.imwrite('__' + get_random_string(8) + '.jpg', face_mat)
        except Exception as e:
            print(e, "Probably out of bounds. Ignore it.")

    if mode == 'shuffle':
        random.shuffle(face_images)
    elif mode == 'reverse':
        face_images.reverse()

    return face_images

def read_simple_inputs(mode):
    target_storage = request.files.get("target")
    source_storage = request.files.get("source")

    target = target_storage.read()
    source = source_storage.read()

    target_faces = extract_faces(target)
    source_faces = extract_faces(source, mode)

    return target, source_faces, target_faces

def do_swap_photo(target, source_faces, target_faces):
    try:
        target_mat = image_to_mat(target)
        source_mats = []
        for source_face in source_faces:
            source_mats.append(image_to_mat(source_face))
        target_faces_mats = []
        for target_face in target_faces:
            target_faces_mats.append(image_to_mat(target_face))

        result_mat = ghost.image_swap(target_mat, source_mats, target_faces_mats)
        result_array = cv2.imencode('.jpg', result_mat)[1]
        result_encoded = np.array(result_array)
        result_byte_encoded = result_encoded.tobytes()

        response = make_response(result_byte_encoded)

        response.headers.set('Content-Type', 'image/jpeg')
        response.headers.set('Content-Disposition', 'attachment', filename='result.jpg')
        return response
    except Exception as e:
        print(e)
        traceback.print_exc()
        return make_response('Error', 500)

@app.route("/ghost-face-swap/api/simple-swap-photo", methods=['POST'])
def simple_swap_photo():
    try:
        mode = request.form.get("mode", None)
        target, source_faces, target_faces = read_simple_inputs(mode)
    except Exception as e:
        print(e)
        traceback.print_exc()
        return make_response('Error', 500)

    return do_swap_photo(target, source_faces, target_faces)

@app.route("/ghost-face-swap/api/flat-swap-photo", methods=['POST'])
def flat_swap_photo():
    try:
        target, source_faces, target_faces = read_flat_inputs()
    except Exception as e:
        print(e)
        traceback.print_exc()
        return make_response('Error', 500)

    return do_swap_photo(target, source_faces, target_faces)


@app.route("/ghost-face-swap/api/swap-photo", methods=['POST'])
def swap_photo():
    try:
        target, source_faces, target_faces = read_inputs()
    except Exception as e:
        print(e)
        traceback.print_exc()
        return make_response('Error', 500)

    return do_swap_photo(target, source_faces, target_faces)


# @app.route("/ghost-face-swap/swap-photo", methods=['POST'])
# def swap_photo():
#     try:
#         target_storage = request.files.get("target")
#         source_storages = request.files.getlist("sources[]")
#         target_faces_storages = request.files.getlist("targetFaces[]")
#
#         target_mat = image_to_mat(target_storage.read())
#         source_mats = []
#         for storage in source_storages:
#             source_mats.append(image_to_mat(storage.read()))
#         target_faces_mats = []
#         for storage in target_faces_storages:
#             target_faces_mats.append(image_to_mat(storage.read()))
#
#         result_mat = ghost.image_swap(target_mat, source_mats, target_faces_mats)
#         result_array = cv2.imencode('.jpg', result_mat)[1]
#         result_encoded = np.array(result_array)
#         result_byte_encoded = result_encoded.tobytes()
#
#         response = make_response(result_byte_encoded)
#
#         response.headers.set('Content-Type', 'image/jpeg')
#         response.headers.set('Content-Disposition', 'attachment', filename='result.jpg')
#         return response
#     except Exception as e:
#         print(e)
#         traceback.print_exc()
#         return make_response('Error', 500)


def do_swap_video(target, source_faces, target_faces):
    try:
        source_mats = []
        for source_face in source_faces:
            source_mats.append(image_to_mat(source_face))
        target_faces_mats = []
        for target_face in target_faces:
            target_faces_mats.append(image_to_mat(target_face))

        result = ghost.video_swap(target, source_mats, target_faces_mats)

        response = make_response(result)

        response.headers.set('Content-Type', 'video/mp4')
        response.headers.set('Content-Disposition', 'attachment', filename='result.mp4')
        return response
    except Exception as e:
        print(e)
        traceback.print_exc()
        return make_response('Error', 500)


@app.route("/ghost-face-swap/api/swap-video", methods=['POST'])
def swap_video():
    try:
        target, source_faces, target_faces = read_inputs()
    except Exception as e:
        print(e)
        traceback.print_exc()
        return make_response('Error', 500)

    return do_swap_video(target, source_faces, target_faces)


@app.route("/ghost-face-swap/api/flat-swap-video", methods=['POST'])
def flat_swap_video():
    try:
        target, source_faces, target_faces = read_flat_inputs()
    except Exception as e:
        print(e)
        traceback.print_exc()
        return make_response('Error', 500)

    return do_swap_video(target, source_faces, target_faces)


# @app.route("/ghost-face-swap/swap-video", methods=['POST'])
# def swap_video():
#     try:
#         target_storage = request.files.get("target")
#         source_storages = request.files.getlist("sources[]")
#         target_faces_storages = request.files.getlist("targetFaces[]")
#
#         source_mats = []
#         for storage in source_storages:
#             source_mats.append(image_to_mat(storage.read()))
#         target_faces_mats = []
#         for storage in target_faces_storages:
#             target_faces_mats.append(image_to_mat(storage.read()))
#
#         result = ghost.video_swap(target_storage.read(), source_mats, target_faces_mats)
#
#         response = make_response(result)
#
#         response.headers.set('Content-Type', 'video/mp4')
#         response.headers.set('Content-Disposition', 'attachment', filename='result.mp4')
#         return response
#     except Exception as e:
#         print(e)
#         traceback.print_exc()
#         return make_response('Error', 500)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "80")))
