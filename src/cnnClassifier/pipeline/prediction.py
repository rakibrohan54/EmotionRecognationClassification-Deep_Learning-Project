import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from PIL import Image  # Importing PIL for image processing

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        # Define class labels based on the correct order (alphabetically)
        self.class_labels = {
            0: 'angry',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'neutral',
            5: 'sad',
            6: 'surprise'
        }

    def convert_to_grayscale(self, img):
        """Convert an RGB image to grayscale."""
        img_gray = img.convert('L')  # Convert to grayscale
        return img_gray

    def predict(self):
        # Load the model
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        # Load the image to be predicted
        imagename = self.filename
        
        # Load the image using PIL to handle RGB to grayscale conversion
        img = Image.open(imagename)
        img = self.convert_to_grayscale(img)  # Convert to grayscale
        
        # Resize the image to the target size (224x224)
        img = img.resize((224, 224))  # Adjust target size for grayscale
        test_image = image.img_to_array(img)
        test_image = np.expand_dims(test_image, axis=0)

        # Normalize the image
        test_image = test_image / 255.0

        # Make the prediction
        result = model.predict(test_image)
        predicted_class = np.argmax(result, axis=1)[0]

        # Map the result to the corresponding class label
        prediction = self.class_labels.get(predicted_class, "Unknown")

        return [{"image": prediction}]
