import os
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

def load_image(file_path):
    img = Image.open(file_path).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    return img_array[np.newaxis, ...]


def classify_image(model, image_path):
    image = load_image(image_path)
    predictions = model.predict(image)[0]

    # Get top 5 predicted class indices and their probabilities
    top5_indices = np.argsort(predictions)[-5:][::-1]
    top5_probs = predictions[top5_indices]

    # Define the range of class indices for dogs (151-268) and cats (281-285)
    dog_classes = list(range(151, 269))
    cat_classes = list(range(281, 286))

    # Check if any of the top 5 predicted classes are dogs or cats with a probability above the threshold
    threshold = 0.5
    for idx, prob in zip(top5_indices, top5_probs):
        if prob > threshold:
            if idx in dog_classes:
                return "dog"
            elif idx in cat_classes:
                return "cat"

    # Return 'unknown' if no dog or cat is detected with a probability above the threshold
    return "unknown"


def main():
    # Load pre-trained MobileNetV2 model for image classification
    model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
    model = tf.keras.Sequential([hub.KerasLayer(model_url, input_shape=(224, 224, 3))])

    # Specify the input directory containing the images
    input_dir = "mix_images"

    # Initialize lists for dog and cat image file names
    dog_images = []
    cat_images = []

    # Iterate through the images in the input directory
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            label = classify_image(model, file_path)

            if label == "dog":
                dog_images.append(file_name)
            elif label == "cat":
                cat_images.append(file_name)

            # MobileNetV2 class indices: 151 - Chihuahua, 281 - tabby cat
            if label == "dog":
                dog_images.append(file_name)
            elif label == "cat":
                cat_images.append(file_name)

    # Save the sorted file names to separate text files
    with open("dog_images.txt", "w") as dog_file:
        for img in dog_images:
            dog_file.write(f"{img}\n")

    with open("cat_images.txt", "w") as cat_file:
        for img in cat_images:
            cat_file.write(f"{img}\n")

    print("Sorting complete. Check dog_images.txt and cat_images.txt for the sorted image file names.")

if __name__ == "__main__":
    main()
