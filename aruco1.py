mport cv2
import numpy as np

# Load the camera matrix and distortion coefficients from a file
camera_matrix = np.array([[785.9682179, 0, 357.16177299], [0,792.81486091, 105.86987584], [0, 0, 1]])
dist_coeffs = np.array([[0.11186678, 0.56365852, -0.06475133, 0.01037753, -1.38199825]])

# Load the image with the Aruco code
img = cv2.imread("image.jpg")

# Detect the Aruco code in the image
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()
corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

# If an Aruco code was detected
if ids is not None:
    # Get the position of the Aruco code in the camera's coordinate system
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], 0.05, camera_matrix, dist_coeffs)
    
    # The known measurement of the Aruco code (in meters)
    code_height = 0.05
    
    # Compute the height of the object using the position of the Aruco code
    fy = camera_matrix[1, 1]
    object_height = code_height * camera_matrix[0, 0] / (tvec[0, 0] * fy)
    print("Object height:", object_height, "m")
else:
    print("Aruco code not detected")