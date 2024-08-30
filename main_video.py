import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load the face cascade classifier
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Provide the path to your .h5 file and video file
model_path = 'Fer2013.h5'
video_path = 'rr.mp4'

# Load the model
model = load_model(model_path)

# Initialize the video capture with the video file
cap = cv2.VideoCapture(video_path)

# Desired frame rate (frames per second)
# frame_rate = 400 # Adjust this value as needed

# Calculate the delay between frames in milliseconds
frame_delay = 100

# Start the video capture loop
while cap.isOpened():
    res, frame = cap.read()
    if not res:
        break
    
    height, width, channel = frame.shape

    # Create an overlay window to write prediction and confidence
    sub_img = frame[0:int(height/6), 0:int(width)]
    black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
    res = cv2.addWeighted(sub_img, 0.77, black_rect, 0.23, 0)

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.8
    FONT_THICKNESS = 2
    label_color = (10, 10, 255)
    label = "Emotion Detection"
    label_dimension = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)[0]
    textX = int((res.shape[1] - label_dimension[0]) / 2)
    textY = int((res.shape[0] + label_dimension[1]) / 2)
    cv2.putText(res, label, (textX, textY), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)

    # Prediction part
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_haar_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    try:
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

            # Draw rectangle around face and display emotion and confidence
            cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            cv2.putText(frame, "Sentiment: {}".format(emotion_prediction), (x, y - 10), FONT, 0.7, label_color, 2)
            label_violation = 'Confidence: {}'.format(str(np.round(np.max(predictions[0]) * 100, 1)) + "%")
            cv2.putText(frame, label_violation, (x, y + h + 20), FONT, 0.7, label_color, 2)
    except:
        pass

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    # Frame rate control
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
