import cv2
import numpy as np
import pandas as pd

# Define a function to detect the dominant color in an image
def detect_color(img):
    # Reshape the image to a 2D array of pixels and 3 color channels
    pixel_values = img.reshape((-1, 3))
    # Convert the pixel values from uint8 to float32
    pixel_values = np.float32(pixel_values)
    # Define the criteria for K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Define the number of clusters (colors) to detect
    k = 3
    # Run K-means clustering
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Convert the centers of each cluster back to uint8
    centers = np.uint8(centers)
    # Get the most common color in the image
    dominant_color = centers[np.argmax(np.unique(labels, return_counts=True)[1])]
    return dominant_color

# Define the video capture device
cap = cv2.VideoCapture(0)

# Define a list to store the detected colors
colors = []

while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()
    # If the frame was successfully read
    if ret:
        # Detect the dominant color in the frame
        color = detect_color(frame)
        # Add the color to the list of detected colors
        colors.append(color)
        # Display the frame and the detected color
        cv2.imshow("Frame", frame)
        cv2.imshow("Color", np.full((100, 100, 3), color.astype('uint8')))
        # If the 'q' key is pressed, exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture device and destroy all windows
cap.release()
cv2.destroyAllWindows()

# Convert the list of detected colors to a NumPy array
colors = np.array(colors)
# Save the colors to a CSV file
df = pd.DataFrame(colors, columns=['R', 'G', 'B'])
df.to_csv('detected_colors.csv', index=False)
