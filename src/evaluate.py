import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("models/plant_disease_model.h5", compile=False)

print("Model output shape:", model.output_shape)
print("Number of classes:", model.output_shape[-1])

# FULL 38 class names (PlantVillage order)
class_names = [
'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
'Blueberry___healthy',
'Cherry___Powdery_mildew','Cherry___healthy',
'Corn___Cercospora_leaf_spot Gray_leaf_spot','Corn___Common_rust','Corn___Northern_Leaf_Blight','Corn___healthy',
'Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
'Orange___Haunglongbing_(Citrus_greening)',
'Peach___Bacterial_spot','Peach___healthy',
'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
'Raspberry___healthy',
'Soybean___healthy',
'Squash___Powdery_mildew',
'Strawberry___Leaf_scorch','Strawberry___healthy',
'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight',
'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
'Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus',
'Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

test_path = "dataset/test"

correct = 0
total = 0

def get_true_label_from_filename(filename):
    name = filename.lower()

    if "applescab" in name:
        return "Apple___Apple_scab"
    elif "corncommonrust" in name:
        return "Corn___Common_rust"
    elif "potatoearlyblight" in name:
        return "Potato___Early_blight"
    elif "potatohealthy" in name:
        return "Potato___healthy"
    elif "tomatoearlyblight" in name:
        return "Tomato___Early_blight"
    elif "tomatohealthy" in name:
        return "Tomato___healthy"
    elif "tomatoyellowcurlvirus" in name:
        return "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
    else:
        return None


for img_name in os.listdir(test_path):

    img_path = os.path.join(test_path, img_name)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]

    true_label = get_true_label_from_filename(img_name)

    print(f"\nImage: {img_name}")
    print("Predicted:", predicted_class)
    print("Actual:", true_label)

    if predicted_class == true_label:
        correct += 1

    total += 1


accuracy = (correct / total) * 100

print("\n==============================")
print(f"Final Test Accuracy: {accuracy:.2f}%")
print("==============================")