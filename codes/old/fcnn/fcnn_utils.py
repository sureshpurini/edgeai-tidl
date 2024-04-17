import onnxruntime
# import cv2
import numpy as np
import tensorflow as tf
import scipy.io
import csv
# from Levenshtein import distance

def levenshtein_accuracy(actual, predicted):
    """
    Calculate accuracy based on Levenshtein distance.

    Parameters:
    - actual (str): The ground truth string.
    - predicted (str): The predicted string.

    Returns:
    - accuracy (float): The Levenshtein accuracy score.
    """
    if len(actual) == 0 and len(predicted) == 0:
        # If both strings are empty, consider accuracy as 1.0
        return 1.0

    edit_distance = distance(actual, predicted)
    max_length = max(len(actual), len(predicted))

    accuracy = 1.0 - (edit_distance / max_length)
    return max(accuracy, 0.0)  # Ensure accuracy is non-negative

def getTestVal(img_name, mat_file="iiit5k/IIIT5K/testCharBound.mat"):
    # Load the MATLAB file
    data = scipy.io.loadmat(mat_file)
    
    # Access the structure (assuming it's named testCharBound)
    test_char_bound = data['testCharBound']  # Adjust the name if needed
    
    # Search for the specified image name in the loaded structure
    # print(test_char_bound[0])
    for entry in test_char_bound[0]:
        # print(entry['ImgName'][0])
        # print(type)
        if entry['ImgName'][0] == img_name:
            # Return the 'chars' value for the found entry
            return entry['chars'][0]
            # pass

    # If image name is not found
    return None
characters = ["0", "1", "2", "3", "4", "5",
                   "6", "7", "8", "9", "A",
                   "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
                   "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
print(characters)

# preprocessing


def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)
    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image


def preprocess_image(image_path, img_size=(128, 32)):

    print(image_path)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    print("processed image size is:",np.shape(image))
    return image


def ctc_decode(pred):
    # greedy decoding
    results = []
    for p in pred:
        decoded_seq = np.argmax(p, axis=1)
        decoded_seq = np.delete(
            decoded_seq, np.where(np.diff(decoded_seq == 0)[0]))
        decoded_seq = decoded_seq[decoded_seq != 38]
        results.append(decoded_seq)
    return results


def num_to_label(num):
    ret = ""
    for ch in num:
        ret += characters[ch-1]
    return ret


def decode_batch_predictions(pred):
    results = ctc_decode(pred)
    results = results[0]
    output_text = num_to_label(results)
    return output_text

