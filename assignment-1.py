import cv2
import numpy as np

def filter_contours(contours):
  filtered_contours = []

  for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 100 or area > 5000:
      continue

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if peri == 0:
      continue

    circularity = (4 * np.pi * area) / (peri * peri)

    if circularity < 0.7 or circularity > 2:
        continue

    print(peri, area, circularity)

    filtered_contours.append(cnt)

  return filtered_contours

def detect_edges_with_laplacian(gray):
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    abs_laplacian = np.uint8(np.absolute(laplacian))
    return abs_laplacian

def detect_edges_with_sobel(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    abs_sobel = np.uint8(np.absolute(sobel_combined))
    return abs_sobel

def apply_morphological_closing(edge_img, kernel_size=10):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, kernel)
    return closed


def segment_coins_circular(image, contours):
    segmented_coins = []
    
    for i, cnt in enumerate(contours):
        # Find the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius) + 5  # Add a small padding
        
        # Create a circular mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        
        # Extract the coin with the mask
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Get the bounding box of the circle for the ROI
        x1 = max(0, int(x - radius))
        y1 = max(0, int(y - radius))
        x2 = min(image.shape[1], int(x + radius))
        y2 = min(image.shape[0], int(y + radius))
        
        # Extract the ROI
        coin_roi = masked_image[y1:y2, x1:x2].copy()
        
        # Store the segmented coin
        segmented_coins.append({
            'index': i,
            'image': coin_roi,
            'center': center,
            'radius': radius,
            'contour': cnt
        })
        
        # Display the segmented coin
        cv2.namedWindow(f"Coin {i}", cv2.WINDOW_NORMAL)
        cv2.imshow(f"Coin {i}", coin_roi)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    return segmented_coins

def process_image(image_path, edge_method="laplacian", segment=True):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found. Please check the image path.")
        return

    # Optionally resize the image if it's too large
    max_width = 800
    max_height = 600
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width/width, max_height/height)
        image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    
    # Create a resizeable window for grayscale
    cv2.namedWindow("Grayscale Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Grayscale Image", 600, 400)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Image", gray)
    cv2.waitKey(0)
    
    if edge_method == "laplacian":
        edges = detect_edges_with_laplacian(gray)
        method_name = "Laplacian Edges"
    elif edge_method == "sobel":
        edges = detect_edges_with_sobel(gray)
        method_name = "Sobel Edges"
    else:
        edges = cv2.Canny(gray, 50, 150)
        method_name = "Canny Edges"
    
    cv2.namedWindow(method_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(method_name, 600, 400)
    cv2.imshow(method_name, edges)
    cv2.waitKey(0)

    ret, binary_edges = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)
    cv2.namedWindow("Binary Edges", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Binary Edges", 600, 400)
    cv2.imshow("Binary Edges", binary_edges)
    cv2.waitKey(0)
    
    closed_edges = apply_morphological_closing(binary_edges, kernel_size=5)
    cv2.namedWindow("Closed Edges", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Closed Edges", 600, 400)
    cv2.imshow("Closed Edges", closed_edges)
    cv2.waitKey(0)
    
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = filter_contours(contours=contours)

    detection_image = image.copy()
    cv2.drawContours(detection_image, valid_contours, -1, (0, 255, 0), 2)
    cv2.namedWindow("Detected Shapes", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detected Shapes", 600, 400)
    cv2.imshow("Detected Shapes", detection_image)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    print(f"Total contours detected using {method_name} and morphological closing: {len(valid_contours)}")

    # ... (existing code) ...
    
    valid_contours = filter_contours(contours=contours)

    detection_image = image.copy()
    cv2.drawContours(detection_image, valid_contours, -1, (0, 255, 0), 2)
    cv2.namedWindow("Detected Shapes", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detected Shapes", 600, 400)
    cv2.imshow("Detected Shapes", detection_image)
    cv2.waitKey(0)
    
    # Add segmentation step
    if segment:
        print("Segmenting coins...")
        segmented_coins = segment_coins_circular(image, valid_contours)
        print(f"Segmented {len(segmented_coins)} coins")
    
    cv2.destroyAllWindows()
    print(f"Total contours detected using {method_name} and morphological closing: {len(valid_contours)}")
    
    if segment:
        return valid_contours, segmented_coins
    else:
        return valid_contours


# def segment_coins(image):
#     segmented_coins = []
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     threshold, threshold_output = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
#     cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Original Image", 600, 400)
#     cv2.imshow("Original Image", image)
#     cv2.waitKey(0)

#     contours, _ = cv2.findContours(threshold_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     valid_contours = filter_contours(contours=contours)

#     for idx, cnt in enumerate(valid_contours):
#         mask = np.zeros_like(gray)
#         cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

#         x, y, w, h = cv2.boundingRect(cnt)

#         cropped = cv2.bitwise_and(gray, gray, mask=mask)

        
#         segmented_coins.append(cropped)
#         cv2.imwrite(f'coin{idx}.jpg', cropped)

#     return segmented_coins

# def segment_coins(img, contours):
#     segmented = []
#     for i, cnt in enumerate(contours):
#         # Create 3-channel color mask
#         mask = np.zeros(img.shape[:2], dtype=np.uint8)
#         cv2.drawContours(mask, [cnt], -1, 255, -1)
        
#         # Apply mask to color image
#         masked = cv2.bitwise_and(img, img, mask=mask)
        
#         # Extract bounding box with original colors
#         x,y,w,h = cv2.boundingRect(cnt)
#         cropped = masked[y:y+h, x:x+w]
        
#         # Create white background instead of black
#         white_bg = np.full_like(cropped, 255)
#         final = np.where(cropped == 0, white_bg, cropped)
        
#         segmented.append(final)
#         cv2.imwrite(f'coin_{i}.png', final)
    
#     return segmented

# Try the different methods by changing the 'edge_method' parameter: "laplacian", "sobel", or "canny".
process_image('photos/IMG_20250301_185845.jpg', edge_method="canny")