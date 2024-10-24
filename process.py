import cv2
import numpy as np

# Load the image
img = cv2.imread('img2.png')

if img is None:
    print("Error: Image not found or could not be loaded.")
    exit()

# Display original image
cv2.imshow("Original Image", img)

# Create a stencil (mask) for the region of interest
stencil = np.zeros_like(img[:, :, 0])

# Define the polygon for the region of interest
polygon = np.array([[50, 570], [380, 360], [560, 360], [780, 570]])

# Fill the polygon area on the stencil with 255 (white)
cv2.fillConvexPoly(stencil, polygon, 255)

# Apply the mask to the image
mask_img = cv2.bitwise_and(img[:, :, 0], img[:, :, 0], mask=stencil)

# Display the masked image
cv2.imshow("Masked Image", mask_img)

# Apply a threshold to the masked image
ret, thresh = cv2.threshold(mask_img, 220, 255, cv2.THRESH_BINARY)

# Display the thresholded image
cv2.imshow('Thresholded Image', thresh)

# Perform Hough Line Transform to detect lines
lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 30, maxLineGap=200)

# Check if lines are detected
if lines is not None:
    # Loop through all detected lines and draw them on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
else:
    print("No lines were detected.")

# Display the final image with detected lines
cv2.imshow("Image with Lines", img)

# Wait for any key press to close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
