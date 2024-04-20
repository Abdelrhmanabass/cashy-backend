import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow as tf
import base64
import cv2
import numpy as np
from flask import Flask,request
from tensorflow.keras.preprocessing import image
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
app = Flask(__name__)
# Load TFLite model
def s() :
    image_path = "F:/first/somthing.jpg" 
    interpreter = tf.lite.Interpreter(model_path="F:/first/FINALbest_float32.tflite")
    interpreter.allocate_tensors()      
# Load and preprocess sample input image
# image_path = "aa.jpg"  # Replace with your sample image path
    image = Image.open(image_path)
    image = image.resize((640, 640))  # Replace width and height with your model's input dimensions
    input_data = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)

# Set input tensor
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
    interpreter.invoke()

# Get output tensor
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
#print("input is",input_details)
#print("out detials is", output_details)

    output_data = tf.nn.softmax(output_data).numpy()  # Apply softmax
    predicted_label_index = np.argmax(output_data)

# Check output (modify as per your model's output format)
    if output_data is not None:
        print("Model made predictions.")
        print("Predictions:", predicted_label_index)
        if 41700 <= predicted_label_index < 41900:  # Check if index falls within the range
         predicted_label = "10EGP"
        elif 50000 <= predicted_label_index < 52000:
         predicted_label = "10EGP_NEW"
        elif 57000 <= predicted_label_index < 59999:
         predicted_label = "100EGP"
        elif 66000 <= predicted_label_index < 68000:
         predicted_label = "200EGP"
        elif 75400 <= predicted_label_index < 76000:
         predicted_label = "20EGP_NEW"
        elif 83000 <= predicted_label_index < 84000:
         predicted_label = "20EGP" 
        elif 92000 <= predicted_label_index < 93000:
         predicted_label = "5EGP"
        elif 100500 <= predicted_label_index < 101000:
         predicted_label = "50EGP"
        else :
         predicted_label = "No Currency detected"   
        
        print("Predicted label:", predicted_label)
        

    else:
        print("Model did not make any predictions.")
    
    return predicted_label


@app.route('/api',methods = ['Put'] )
def index():
    inputchar = request.get_data()
    imgdata = base64.b64decode(inputchar)
    filename = 'somthing.jpg'  
    with open(filename, 'wb') as f:
      f.write(imgdata)
    o=s()
    return o
if __name__ == "__main__":
    app.run(debug=True)
