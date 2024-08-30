import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Load the face cascade classifier
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the model
model = load_model('Fer2013.h5')

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
    lable_color = (10, 10, 255)
    lable = "Emotion Detection made by RP"
    lable_dimension = cv2.getTextSize(lable, FONT, FONT_SCALE, FONT_THICKNESS)[0]
    textX = int((res.shape[1] - lable_dimension[0]) / 2)
    textY = int((res.shape[0] + lable_dimension[1]) / 2)
    cv2.putText(res, lable, (textX, textY), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)

    # Prediction part
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,  # Adjust to avoid detecting surrounding objects
        minNeighbors=5,   # Increase to reduce false positives
        minSize=(30, 30)  # Minimum face size to detect
    )

    try:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = gray_image[y-5:y+h+5, x-5:x+w+5]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            image_pixels = img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis=0)
            image_pixels /= 255

            predictions = model.predict(image_pixels)
            max_index = np.argmax(predictions[0])
            emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            emotion_prediction = emotion_detection[max_index]
            cv2.putText(res, "Sentiment: {}".format(emotion_prediction), (0, textY + 22 + 5), FONT, 0.7, lable_color, 2)
            lable_violation = 'Confidence: {}'.format(str(np.round(np.max(predictions[0]) * 100, 1)) + "%")
            violation_text_dimension = cv2.getTextSize(lable_violation, FONT, FONT_SCALE, FONT_THICKNESS)[0]
            violation_x_axis = int(res.shape[1] - violation_text_dimension[0])
            cv2.putText(res, lable_violation, (violation_x_axis, textY + 22 + 5), FONT, 0.7, lable_color, 2)
    except:
        pass

    # Overlay the results on the frame
    frame[0:int(height/6), 0:int(width)] = res

    # Display the frame
    cv2.imshow('frame', frame)

    # Press 'q' to quit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
