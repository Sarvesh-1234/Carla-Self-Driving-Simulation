import cv2
import numpy as np

def region_of_interest(img, vertices):
    # Create a mask the same size as the image, filled with zeros (black)
    mask = np.zeros_like(img)
    
    # Fill the polygon defined by "vertices" with white (255)
    cv2.fillPoly(mask, vertices, 255)
    
    # Mask the image using bitwise AND
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=(255, 0, 0), thickness=10):
    # Create an empty image to draw lines on
    line_img = np.zeros_like(img)
    
    # If no lines are detected, return the original image
    if lines is None:
        return img

    # Draw each detected line
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

    # Combine the line image with the original image
    return cv2.addWeighted(img, 0.8, line_img, 1, 0)

def process_image(image, canny_low_thresh=50, canny_high_thresh=150,
                  blur_kernel_size=5, rho=1, theta=np.pi/180, hough_threshold=30,
                  min_line_length=100, max_line_gap=200):
    
    # 1. Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply Gaussian blur to smooth the image
    blur = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
    
    # 3. Use Canny Edge Detection with dynamic thresholds
    v = np.median(blur)
    lower_thresh = int(max(0, 0.66 * v))  # Adjust these thresholds
    upper_thresh = int(min(255, 1.33 * v))
    edges = cv2.Canny(blur, lower_thresh, upper_thresh)
    
    # 4. Define a larger region of interest (ROI) to cover more of the lanes
    height, width = image.shape[:2]
    region_of_interest_vertices = np.array([[
        (50, height), 
        (width // 2 - 200, height // 2 + 100),  # Adjusted for better lane coverage
        (width // 2 + 200, height // 2 + 100), 
        (width - 50, height)
    ]], dtype=np.int32)
    
    # 5. Apply the ROI mask to the edges
    cropped_edges = region_of_interest(edges, region_of_interest_vertices)
    
    # 6. Use Hough Line Transform to detect lane lines
    lines = cv2.HoughLinesP(cropped_edges, rho=rho, theta=theta, 
                            threshold=hough_threshold, minLineLength=min_line_length, 
                            maxLineGap=max_line_gap)
    
    # 7. Draw detected lines on the original image
    line_image = draw_lines(image, lines, thickness=8)  # Reduced thickness for better clarity
    
    return line_image

# Load the image
image = cv2.imread('img1.png')

if image is None:
    print("Error: Image not found or could not be loaded.")
    exit()

# Process the image to detect lanes
processed_image = process_image(image)

# Display the final output
cv2.imshow('Lane Detection', processed_image)

# Wait for any key press to close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
