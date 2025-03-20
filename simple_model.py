import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # MobileNetV2 specific preprocessing

# Load a pretrained MobileNetV2 model (without top layers)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers for your specific problem (optional)
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global average pooling to reduce dimensions
x = Dense(1024, activation='relu')(x)  # Fully connected layer
x = Dropout(0.5)(x)  # Dropout layer to prevent overfitting
x = BatchNormalization()(x)  # Batch normalization for better performance
predictions = Dense(1, activation='sigmoid')(x)  # Final output layer for binary classification

# Create the model with the base model and the custom layers
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all layers of the pretrained MobileNetV2 model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model (no training, just for inference)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Save the model (optional)
MODEL_PATH = 'vio_vedio.h5'
model.save(MODEL_PATH)

print("Pretrained MobileNetV2 model loaded and saved successfully!")

# Image preprocessing for prediction
def predict_image(img_path):
    """Make predictions on a single image."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image for MobileNetV2

    # Prediction
    prediction = model.predict(img_array)
    
    # Adjust the threshold if necessary
    if prediction >= 0.5:
        return "Violence detected"
    else:
        return "No violence detected"

# Example of using the function:
# result = predict_image("path_to_your_image.jpg")
# print(result)
