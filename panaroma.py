import cv2
import matplotlib.pyplot as plt

def create_panorama2(images_path):
  if len(images_path) < 2:
    print("At least 2 images are required to perform panorama stitching")
    return None

  print('Loaded images for panorama stitching')

  orb = cv2.ORB_create(nfeatures=4000)

  images = [cv2.imread(img) for img in images_path]

  for i, img in enumerate(images):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    img_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(f'keypoints_{i}.jpg', img_keypoints)
    plt.figure(figsize=(10, 8))
    plt.title(f'Keypoints for image {i}')
    plt.imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    print(f'Keypoints for image {i} saved as keypoints_{i}.jpg')


  stitcher = cv2.Stitcher_create()
  status, panorama = stitcher.stitch(images)

  if status == cv2.Stitcher_OK:
    print('Panorama stitching successful')
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Panorama')
    plt.show()
    cv2.imwrite('panorama.jpg', panorama)
  else:
    print('Panorama stitching failed')
    return None
