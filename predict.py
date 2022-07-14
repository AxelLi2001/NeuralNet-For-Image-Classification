import argparse
import tensorflow as tf
import json
import numpy as np
from PIL import Image
import tensorflow_hub as hub

if __name__ == '__main__':
    # initialize
    parser = argparse.ArgumentParser(
        description = 'Image Recognizer')
    # add argument
    parser.add_argument('path', help = "the path to image file")
    parser.add_argument('model', help = "the model's path")
    parser.add_argument('--top_k', help = "return k most likely classes", type = int, default = 1)
    parser.add_argument('--category_names', help = "path to json file mapping labels to flower names", default = "")
    args = parser.parse_args()
    
    ## Make prediction
    
    # load model
    model = tf.keras.models.load_model(
       args.model,
       custom_objects={'KerasLayer':hub.KerasLayer})
    # load image
    img = np.asarray(Image.open(args.path))
    
    # process image
    SIZE = 224
    tensor_img = tf.convert_to_tensor(img, dtype=tf.float32)
    resized_img = tf.image.resize(tensor_img, (SIZE, SIZE)).numpy()
    processed_img = resized_img / 255
    
    # prediction
    prediction = model.predict(np.expand_dims(processed_img, axis = 0))
    pred_index_descending = np.flip(np.argsort(prediction[0]))
    
    # prepare result
    classes = []
    probs = []
    for i in range(0, args.top_k):
        probs.append(round(prediction[0][pred_index_descending[i]], 3))
        classes.append(pred_index_descending[i])
        
    if args.category_names != "":
        # load dictionary
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        for i in range(0, args.top_k):
            classes[i] = class_names[str((classes[i] + 1))]
        
        
    # give result
    print("The image was identified as label: ", classes)
    print("With probabilities: ", probs)
    
