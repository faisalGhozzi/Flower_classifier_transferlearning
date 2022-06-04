"""This is the main entrypoint of this project.
    When executed as
    python main.py input_folder/
    it should perform the neural network inference on all images in the `input_folder/` and print the results.
"""

from argparse import ArgumentParser
import cv2
import numpy as np
import tensorflow as tf
import sys
import os
import glob
from urllib.request import urlopen
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
ext = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
def parse_args():
    """Define CLI arguments and return them.

    Feel free to modify this function if needed.

    Returns:
        Namespace: parser arguments
    """
    parser = ArgumentParser(description="This application helps with detecting whether a flower is a daisy, dandelion, rose, sunflower or tulip")
    parser.add_argument("-i", "--input_folder", type=str, metavar='', help="A folder with images to analyze.")
    parser.add_argument("-u", "--url", type=str, metavar='', help='A link to an image to analyze.')
    parser.add_argument("-p", "--match_pattern", type=str, metavar='', help='Get files similar to this pattern')
    args = parser.parse_args()
    return args

def get_images_in_folder(folder, match_pattern = None):
    """Fetches images in a folder, if match_pattern is specified without an extention
    It matches the pattern while testing with availabe image files extentions

    Args:
        folder (str): Path to the folder containing the images
        match_pattern (str, optional): Pattern to match while fetching images. Defaults to None.

    Returns:
        list: List of images paths
    """
    if match_pattern == None:
        files = [item for i in [glob.glob(folder+'/*.%s' % ext) for ext in ["jpg","jpeg","png","gif","bmp"]] for item in i]
    else:
        if match_pattern.lower().endswith(ext):
            files = [item for i in [glob.glob(folder+'/'+match_pattern)] for item in i]
        else:
            files = [item for i in [glob.glob(folder+'/'+match_pattern+'.%s' % ext) for ext in ["jpg","jpeg","png","gif","bmp"]] for item in i]
    if len(files) == 0:
        print('No Pictures Found')
        quit()
    return files

def get_image_from_link(url):
    """Extract image from a given link

    Args:
        url (str): Link containing the image

    Returns:
        numpy.ndarray: Image as a numpy.ndarray
    """
    response = urlopen(url)
    img_array = np.array(bytearray(response.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)
    return img
    
def resize_image(image, url, IMG_SIZE):
    """Function that resizes an image using opencv

    Args:
        image (str): Path to image

    Returns:
        numpy.array: Resized image
    """
    if url != None:
        resized_img = cv2.resize(image, (IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    else:
        img = cv2.imread(image)
        resized_img = cv2.resize(img, (IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return resized_img
    
def convert_and_prepare(image):
    """Converts an image to a numpy.ndarray while setting it's values type as a float32
    and expanding it's dimensions so it can be processed by the predictive model

    Args:
        image (numpy.ndarray): Image to be preprocessed

    Returns:
        numpy.ndarray: Prepared image
    """
    img = np.asarray(image)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(image):
    """Predicts an image

    Args:
        image (numpy.ndarray): Image to be predicted

    Returns:
       numpy.ndarray: Prediction score
       numpy.ndarray: Predicted classes
    """
    model = load_model('./custom_model')
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    output_data = model.predict(image)
    return output_data, output_data

def prediction(image, url=None):
    """Predict function that takes in a model type and predict using it

    Args:
        image (str): Path to the image
        url (str, optional): Specifies if the file is an url. Defaults to None.

    Returns:
        tensorflow.python.framework.ops.EagerTensor: Prediction score
        numpy.ndarray: Predicted classes
    """
    IMG_SIZE = 224
    img = resize_image(image, url, IMG_SIZE)
    img = convert_and_prepare(img)
    return predict_image(img)

def predict_from_link(url):
    """Predict image from link and show score

    Args:
        url (str): Link to the image
    """
    image = get_image_from_link(url)
    score, output = prediction(image, True)
    print({url:{"score":"{:.2f}".format(np.max(score)), "class":class_names[np.argmax(output[0])]}})

def predict_from_folder(folder, match_pattern=None):
    """Predict image(s) from folder and show score(s)

    Args:
        folder (str): Folder containing images
        match_pattern (str, optional): Pattern to match while fetching images. Defaults to None.
    """
    images = get_images_in_folder(folder, match_pattern)
    for img in images:
        score, output = prediction(img)
        print({os.path.basename(img):{"score":"{:.2f}".format(np.max(score)), "class":class_names[np.argmax(output[0])]}})

if __name__ == "__main__": 
    cli_args = parse_args()
    if cli_args.input_folder == None and cli_args.url == None:
        print('Please make sure to input a folder or a url to analyze "-h or --help for more information"', file=sys.stderr)
    elif cli_args.input_folder != None and cli_args.url == None:
        print(f"Analyzing folder : {cli_args.input_folder}", file=sys.stderr)
        predict_from_folder(cli_args.input_folder, cli_args.match_pattern)
    elif cli_args.input_folder == None and cli_args.url != None:
        print(f"Analyzing url : {cli_args.url}", file=sys.stderr)
        predict_from_link(cli_args.url)
    else:
        print(f"Analyzing folder : {cli_args.input_folder}", file=sys.stderr)
        print(f"Analyzing url : {cli_args.url}", file=sys.stderr)
        predict_from_folder(cli_args.input_folder, cli_args.match_pattern)
        predict_from_link(cli_args.url)