from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np

IMG_SIZE = (128, 128)

def preprocess_image(img_file):
    img = image.load_img(img_file, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # 🔥 CRUCIAL
    return img, img_array