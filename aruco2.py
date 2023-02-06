import cv2
import numpy as np

# Load the camera matrix and distortion coefficients from a file
camera_matrix = np.array([[628.1743, 0, 324.8749], [0,627.3089, 239.2556], [0, 0, 1]])
dist_coeffs = np.array([[0.0182, -0.0873, 0, 0]])

# The known measurement of the Aruco code (in meters)
code_height = 92.33/1000

# Open a video capture object
cap = cv2.VideoCapture(2)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

while True:
    # Capture a frame from the video
    ret, img = cap.read()
    # img = cv2.resize(img, (176, 144)) 

    # Detect the Aruco code in the image
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    parameters = cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

    # If an Aruco code was detected
    if ids is not None:
        # Draw the Aruco code on the image
        cv2.aruco.drawDetectedMarkers(img, corners, ids)

        # Get the position of the Aruco code in the camera's coordinate system
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], 0.05, camera_matrix, dist_coeffs)
        print(tvec.shape)

        # Compute the height of the object using the position of the Aruco code
        fy = camera_matrix[1, 1]
        object_height = code_height * camera_matrix[0, 0] / (tvec[0, 0] * fy)
        print("Object height:", object_height, "m")

        # Draw the estimated height on the image
        # cv2.putText(img, "Object height: {:.2f} m".format(object_height), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the image
    cv2.imshow("Aruco Code Detection", img)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()