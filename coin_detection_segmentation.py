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
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius) + 5  
        
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)

        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        x1 = max(0, int(x - radius))
        y1 = max(0, int(y - radius))
        x2 = min(image.shape[1], int(x + radius))
        y2 = min(image.shape[0], int(y + radius))
        
        coin_roi = masked_image[y1:y2, x1:x2].copy()

        cv2.imwrite(f'output/coins/coin-{i}.jpg', coin_roi)
        
        segmented_coins.append({
            'index': i,
            'image': coin_roi,
            'center': center,
            'radius': radius,
            'contour': cnt
        })
        
        # cv2.namedWindow(f"Coin {i}", cv2.WINDOW_NORMAL)
        # cv2.imshow(f"Coin {i}", coin_roi)
        # cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    return segmented_coins

def process_image(image_path, edge_method="laplacian", segment=True):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found. Please check the image path.")
        return

    max_width = 800
    max_height = 600
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width/width, max_height/height)
        image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    
    # cv2.namedWindow("Grayscale Image", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Grayscale Image", 600, 400)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Grayscale Image", gray)
    # cv2.waitKey(0)
    
    if edge_method == "laplacian":
        edges = detect_edges_with_laplacian(gray)
        method_name = "Laplacian Edges"
    elif edge_method == "sobel":
        edges = detect_edges_with_sobel(gray)
        method_name = "Sobel Edges"
    else:
        edges = cv2.Canny(gray, 50, 150)
        method_name = "Canny Edges"
    
    # cv2.namedWindow(method_name, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(method_name, 600, 400)
    # cv2.imshow(method_name, edges)
    # cv2.waitKey(0)

    ret, binary_edges = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)
    # cv2.namedWindow("Binary Edges", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Binary Edges", 600, 400)
    # cv2.imshow("Binary Edges", binary_edges)
    # cv2.waitKey(0)
    
    closed_edges = apply_morphological_closing(binary_edges, kernel_size=5)
    # cv2.namedWindow("Closed Edges", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Closed Edges", 600, 400)
    # cv2.imshow("Closed Edges", closed_edges)
    # cv2.waitKey(0)
    
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = filter_contours(contours=contours)

    detection_image = image.copy()
    cv2.drawContours(detection_image, valid_contours, -1, (0, 255, 0), 2)
    cv2.namedWindow("Detected Coins", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detected Coins", 600, 400)
    cv2.imshow("Detected Coins", detection_image)
    cv2.waitKey(0)
    
    
    if segment:
        print("Segmenting coins...")
        segmented_coins = segment_coins_circular(image, valid_contours)
    
    cv2.destroyAllWindows()
    print(f"Total coins detected using {method_name}: {len(valid_contours)}")
    
    if segment:
        return valid_contours, segmented_coins
    else:
        return valid_contours

if __name__ == "__main__":
    # Different 'edge_method' parameters: "laplacian", "sobel", or "canny".
    process_image('photos/coins/coins-1.jpg', edge_method="canny")