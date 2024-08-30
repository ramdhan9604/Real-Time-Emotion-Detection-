import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load the face cascade classifier
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Provide the path to your .h5 file and image file
model_path = 'Fer2013.h5'
image_path = 'bb.png'

# Load the model
model = load_model(model_path)

# Load the image
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_haar_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Define parameters for text overlay
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5  # Smaller font scale
FONT_THICKNESS = 1  # Thinner font
label_color = (10, 10, 255)
background_color = (0, 0, 0)  # Background color for text

# Process each face detected
for (x, y, w, h) in faces:
    # Extract face region and preprocess
    roi_gray = gray_image[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    image_pixels = img_to_array(roi_gray)
    image_pixels = np.expand_dims(image_pixels, axis=0)
    image_pixels /= 255

    # Predict emotion
    predictions = model.predict(image_pixels)
    max_index = np.argmax(predictions[0])
    emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    emotion_prediction = emotion_detection[max_index]

    # Draw rectangle around face
    cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)

    # Prepare text for sentiment and confidence
    sentiment_text = "Sentiment: {}".format(emotion_prediction)
    confidence_text = 'Confidence: {:.1f}%'.format(np.round(np.max(predictions[0]) * 100, 1))

    # Calculate text size and positions
    sentiment_text_size = cv2.getTextSize(sentiment_text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
    confidence_text_size = cv2.getTextSize(confidence_text, FONT, FONT_SCALE, FONT_THICKNESS)[0]

    # Create background rectangles for text
    cv2.rectangle(image, (x, y - 30), (x + sentiment_text_size[0] + 10, y - 30 + sentiment_text_size[1] + 10), background_color, -1)
    cv2.rectangle(image, (x, y + h + 10), (x + confidence_text_size[0] + 10, y + h + 10 + confidence_text_size[1] + 10), background_color, -1)

    # Put text on the image
    cv2.putText(image, sentiment_text, (x + 5, y - 5), FONT, FONT_SCALE, label_color, FONT_THICKNESS)
    cv2.putText(image, confidence_text, (x + 5, y + h + 25), FONT, FONT_SCALE, label_color, FONT_THICKNESS)

# Display the image with detections
cv2.imshow('Emotion Detection', image)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()