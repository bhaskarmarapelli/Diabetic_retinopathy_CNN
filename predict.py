import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt


def predict_class(path, model_path):
    # Load the image
    img = cv2.imread(path)

    # Convert image to RGB
    RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image to the model's input size
    RGBImg = cv2.resize(RGBImg, (224, 224))

    # Normalize the image
    image = np.array(RGBImg) / 255.0

    # Load the model
    new_model = tf.keras.models.load_model(model_path)

    # Make a prediction
    prediction = new_model.predict(np.array([image]))

    # Get the predicted class
    predicted_class = np.argmax(prediction, axis=1)

    # Display the image
    plt.imshow(RGBImg)

    # Define class labels based on your specific classes
    class_labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']

    # Display the predicted class
    predicted_label = class_labels[predicted_class[0]]
    print(f'Predicted Class: {predicted_label}')
    plt.title(f'Predicted Class: {predicted_label}')

    plt.show()


# Example usage
predict_class('path_to_your_image.jpg', 'path_to_your_model.h5')

# Example usage
predict_class('C:/Users/Bhaskar Marapelli/Downloads/gaussian_filtered_images/gaussian_filtered_images/Mild/0024cdab0c1e.png')
predict_class('C:/Users/Bhaskar Marapelli/Downloads/gaussian_filtered_images/gaussian_filtered_images/No_DR/2ef10194e80d.png')
