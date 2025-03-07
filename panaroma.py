# Import necessary libraries
import cv2  # OpenCV library for computer vision and image processing

# Import custom utility function to save images
from utils import save_image


def create_panorama(images_path):
  """
  Creates a panorama by stitching multiple images together.
  
  Args:
      images_path: List of file paths to the images to be stitched
      
  Returns:
      Panorama image if successful, None otherwise
  """
  # Check if we have enough images for stitching
  if len(images_path) < 2:
    print("At least 2 images are required to perform panorama stitching")
    return None

  print('Loaded images for panorama stitching')

  # Create ORB (Oriented FAST and Rotated BRIEF) feature detector
  # nfeatures=4000 specifies the maximum number of features to detect
  orb = cv2.ORB_create(nfeatures=4000)

  # Load all images from the provided paths
  images = [cv2.imread(img) for img in images_path]

  # Process each image to detect and visualize keypoints
  for i, img in enumerate(images):
    # Convert to grayscale for better feature detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    # Draw keypoints on the image for visualization
    # DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS shows keypoints with size and orientation
    img_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Save the image with keypoints highlighted
    save_image(f'output/panorama/keypoints_{i}.jpg', img_keypoints)

  # Create a stitcher object to combine the images
  stitcher = cv2.Stitcher_create()
  
  # Attempt to stitch the images together
  # status: result code indicating success or failure
  # panorama: the resulting stitched image
  status, panorama = stitcher.stitch(images)

  # Check if stitching was successful
  if status == cv2.Stitcher_OK:
    print('Panorama stitching successful')
    # Save the resulting panorama
    save_image('output/panorama/panorama.jpg', panorama)
    return panorama
  else:
    print('Panorama stitching failed')
    return None
  

# Execute code only when run directly (not when imported)
if __name__ == '__main__':
  # Define paths to images that should be stitched together
  # These should be sequential images with overlapping content
  image_paths = [
    'photos/panorama/Room-01.jpg',
    'photos/panorama/Room-02.jpg',
    'photos/panorama/Room-03.jpg',
    'photos/panorama/Room-04.jpg'
  ]

  # Call the function to create and save the panorama
  create_panorama(image_paths)