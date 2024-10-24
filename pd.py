import cv2
import numpy as np

def region_of_interest(img, vertices):
    # Create a mask that matches the input image dimensions
    mask = np.zeros_like(img)
    
    # Fill the region inside the polygon with white
    cv2.fillPoly(mask, vertices, 255)
    
    # Apply the mask to the image
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=(255, 0, 0), thickness=10):
    line_img = np.zeros_like(img)
    
    if lines is None:
        return img

    # Iterate over the detected lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

    # Combine the original image with the lines
    return cv2.addWeighted(img, 0.8, line_img, 1, 0)

def process_image(image, canny_low_thresh=50, canny_high_thresh=150,
                  blur_kernel_size=5, rho=1, theta=np.pi/180, hough_threshold=50,
                  min_line_length=100, max_line_gap=50):
    
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply Gaussian Blur to smooth the image
    blur = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
    
    # 3. Use Canny Edge Detection with dynamic thresholds
    v = np.median(blur)
    sigma = 0.33
    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(blur, lower_thresh, upper_thresh)
    
    # 4. Define the region of interest (ROI) for the lane
    height, width = image.shape[:2]
    region_of_interest_vertices = np.array([[
        (50, height), (width // 2 - 50, height // 2 + 50), 
        (width // 2 + 50, height // 2 + 50), (width - 50, height)
    ]], dtype=np.int32)
    
    # 5. Apply the mask to the edges
    cropped_edges = region_of_interest(edges, region_of_interest_vertices)
    
    # 6. Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(cropped_edges, rho=rho, theta=theta, 
                            threshold=hough_threshold, minLineLength=min_line_length, 
                            maxLineGap=max_line_gap)
    
    # 7. Draw detected lines on the original image
    line_image = draw_lines(image, lines)
    
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
