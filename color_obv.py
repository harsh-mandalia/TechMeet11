import cv2
import numpy as np

# Define the lower and upper bounds for the red color
lower_red = np.array([136, 87, 111])
upper_red = np.array([180, 255, 255])

# Open the video capture object
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Read each frame from the video stream
    ret, frame = cap.read()

    # Convert the frame from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)

        # Get the bounding rect of the largest contour
        x, y, w, h = cv2.boundingRect(c)

        # Draw the bounding box around the red color
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Get the center of the bounding box
        center = (x + w // 2, y + h // 2)

        # Draw the center of the bounding box
        cv2.circle(frame, center, 3, (0, 0, 255), -1)

        # Put the center coordinates on the bounding box
        text = "({}, {})".format(center[0], center[1])
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Show the resulting frame
    cv2.imshow("Frame", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()