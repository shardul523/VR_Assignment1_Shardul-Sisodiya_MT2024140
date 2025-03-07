import sys
import cv2
import numpy as np
from utils import save_image

def filter_contours(contours):
    """
    Filter contours based on area, perimeter, and circularity criteria.
    
    Args:
        contours: List of contour arrays from cv2.findContours()
        
    Returns:
        filtered_contours: List of contours that meet all filtering criteria
    """
    # Initialize empty list to store filtered contours
    filtered_contours = []

    # Iterate through each contour
    for cnt in contours:
        # Calculate contour area
        area = cv2.contourArea(cnt)
        # Skip contours that are too small or too large
        if area < 100 or area > 5000:
            continue

        # Calculate perimeter of closed contour
        peri = cv2.arcLength(cnt, True)
        # Approximate contour shape with simpler polygon
        # 0.02 * perimeter is the approximation accuracy parameter
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # Skip contours with zero perimeter to avoid division by zero
        if peri == 0:
            continue

        # Calculate circularity: (4 * π * area) / (perimeter²)
        # Perfect circle has circularity of 1.0
        circularity = (4 * np.pi * area) / (peri * peri)

        # Skip contours that aren't circular enough
        # Values between 0.7 and 2 represent approximately circular shapes
        if circularity < 0.7 or circularity > 2:
            continue

        # Add contour to filtered list if it passes all criteria
        filtered_contours.append(cnt)

    # Return the filtered contours
    return filtered_contours


def detect_edges_with_laplacian(gray):
    """
    Detect edges in a grayscale image using the Laplacian operator.
    
    Args:
        gray: Grayscale input image (numpy array)
        
    Returns:
        abs_laplacian: Edge-detected image with Laplacian operator
    """
    # Apply Laplacian operator for edge detection
    # cv2.CV_64F specifies 64-bit float output to capture negative transitions
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Convert to absolute values to handle both "dark-to-light" and "light-to-dark" edges
    # Laplacian produces negative values at some edges and positive at others
    abs_laplacian = np.uint8(np.absolute(laplacian))
    
    # Return the edge-detected image converted to 8-bit format suitable for display
    return abs_laplacian


def detect_edges_with_sobel(gray):
    """
    Detect edges in a grayscale image using the Sobel operator.
    
    Args:
        gray: Grayscale input image (numpy array)
        
    Returns:
        abs_sobel: Edge-detected image using combined Sobel operators
    """
    # Apply Sobel operator in x direction (horizontal edges)
    # Parameters: image, output depth, x derivative order (1), y derivative order (0), kernel size (3x3)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    
    # Apply Sobel operator in y direction (vertical edges)
    # Parameters: image, output depth, x derivative order (0), y derivative order (1), kernel size (3x3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude by combining x and y gradients
    # This gives the overall edge strength at each pixel
    sobel_combined = cv2.magnitude(sobelx, sobely)
    
    # Convert to absolute values and 8-bit format for display
    abs_sobel = np.uint8(np.absolute(sobel_combined))
    
    # Return the final edge-detected image
    return abs_sobel


def apply_morphological_closing(edge_img, kernel_size=10):
    """
    Apply morphological closing operation to an edge image to connect nearby edges 
    and fill small gaps.
    
    Args:
        edge_img: Input edge image (typically from edge detection)
        kernel_size: Size of the square kernel used for the morphological operation (default: 10)
        
    Returns:
        closed: Image after applying morphological closing
    """
    # Create a rectangular structuring element (kernel) of specified size
    # MORPH_RECT creates a square/rectangular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Apply morphological closing operation (dilation followed by erosion)
    # This helps connect nearby edges and close small gaps
    closed = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, kernel)
    
    # Return the processed image
    return closed


def segment_coins_circular(image, contours):
    """
    Segment individual coins from an image using circular contours.
    
    Args:
        image: Original input image
        contours: List of contours representing potential coins
        
    Returns:
        segmented_coins: List of dictionaries containing information about each segmented coin
    """
    # Initialize empty list to store segmented coin information
    segmented_coins = []
    
    # Process each contour
    for i, cnt in enumerate(contours):
        # Find minimum enclosing circle for the contour
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        # Add 5 pixels to radius to ensure the entire coin is captured
        radius = int(radius) + 5  
        
        # Create a blank mask the same size as the input image
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # Draw a filled white circle on the mask
        # The -1 parameter makes the circle filled rather than just an outline
        cv2.circle(mask, center, radius, 255, -1)

        # Apply the mask to the original image to isolate the coin
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Calculate bounding box coordinates for the coin region
        # Ensure coordinates stay within image boundaries
        x1 = max(0, int(x - radius))
        y1 = max(0, int(y - radius))
        x2 = min(image.shape[1], int(x + radius))
        y2 = min(image.shape[0], int(y + radius))
        
        # Extract the region of interest (ROI) containing the coin
        coin_roi = masked_image[y1:y2, x1:x2].copy()

        # Save the extracted coin image to a file
        save_image(f'output/coins/coin-{i}.jpg', coin_roi)
        
        # Store coin information in a dictionary
        segmented_coins.append({
            'index': i,            # Unique identifier for the coin
            'image': coin_roi,     # Extracted image of the coin
            'center': center,      # Center coordinates of the coin
            'radius': radius,      # Radius of the coin
            'contour': cnt         # Original contour points
        })
    
    # Return the list of segmented coin information
    return segmented_coins


def process_image(image_path, edge_method="laplacian", segment=True):
    """
    Process an image to detect and optionally segment coins.
    
    Args:
        image_path: Path to the input image
        edge_method: Edge detection method to use ("laplacian", "sobel", or "canny" default)
        segment: Whether to segment individual coins (default: True)
        
    Returns:
        valid_contours: List of filtered contours representing coins
        segmented_coins: List of segmented coin data (if segment=True)
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found. Please check the image path.")
        return

    # Resize image if it's too large while maintaining aspect ratio
    max_width = 800
    max_height = 600
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width/width, max_height/height)
        image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply the selected edge detection method
    if edge_method == "laplacian":
        edges = detect_edges_with_laplacian(gray)
        method_name = "Laplacian Edges"
    elif edge_method == "sobel":
        edges = detect_edges_with_sobel(gray)
        method_name = "Sobel Edges"
    else:
        # Default to Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        method_name = "Canny Edges"
    
    # Threshold the edge image to create binary image
    ret, binary_edges = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)
    
    # Apply morphological closing to connect broken edges
    closed_edges = apply_morphological_closing(binary_edges, kernel_size=5)
    
    # Find contours in the processed edge image
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to identify potential coins
    valid_contours = filter_contours(contours=contours)

    # Create visualization of detected coins
    detection_image = image.copy()
    cv2.drawContours(detection_image, valid_contours, -1, (0, 255, 0), 2)
    
    # Display the result
    cv2.namedWindow("Detected Coins", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detected Coins", 600, 400)
    cv2.imshow("Detected Coins", detection_image)
    cv2.waitKey(0)  # Wait for a key press

    # Save the result image
    save_image('./output/coins/coins_detected.jpg', detection_image)
    
    # Segment coins if requested
    if segment:
        print("Segmenting coins...")
        segmented_coins = segment_coins_circular(image, valid_contours)
    
    # Clean up display windows
    cv2.destroyAllWindows()
    
    # Print summary of results
    print(f"Total coins detected using {method_name}: {len(valid_contours)}")
    
    # Return appropriate results
    if segment:
        return valid_contours, segmented_coins
    else:
        return valid_contours


if __name__ == "__main__":
    """
    Main execution block that runs when the script is executed directly.
    
    This allows the coin detection pipeline to be run from the command line.
    Usage: python script_name.py [optional_image_path]
    
    The script can process images using different edge detection methods:
    - "laplacian": Second-order derivative method
    - "sobel": First-order derivative method in multiple directions
    - "canny": Multi-stage algorithm (default in this implementation)
    """
    # Set default image path
    image_path = 'photos/coins/coins-1.jpg'
    
    # Override default image path if provided as command-line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    # Process the image using the Canny edge detection method
    # Note: Other available methods are "laplacian" and "sobel"
    process_image(image_path, edge_method="canny")