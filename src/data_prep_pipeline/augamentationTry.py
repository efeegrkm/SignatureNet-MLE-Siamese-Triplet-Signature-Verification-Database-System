import cv2
import numpy as np
def rotate_image(img, angle_deg=1):
    """
    img: grayscale numpy array
    angle_deg: döndürme açısı (pozitif saat yönü)
    """
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle_deg, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows), borderValue=0)
    return rotated
def translate_image(img, tx=5, ty=5):
    """
    tx: x yönünde piksel kaydırma
    ty: y yönünde piksel kaydırma
    """
    rows, cols = img.shape
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(img, M, (cols, rows), borderValue=0)
    return translated
def scale_image(img, scale=0.7):
    """
    scale: imzanın boyutu (1 = orijinal)
    """
    rows, cols = img.shape
    resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # Tekrar orijinal boyuta döndür
    resized = cv2.resize(resized, (cols, rows), interpolation=cv2.INTER_LINEAR)
    return resized
def add_gaussian_noise(img, sigma=20):
    """
    sigma: gürültü standard deviation
    """
    noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
    noisy = img.astype(np.int16) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy
def dilate_erode_image(img, mode='dilate', ksize=2):
    """
    mode: 'dilate' veya 'erode'
    ksize: kernel boyutu
    """
    kernel = np.ones((ksize, ksize), np.uint8)
    if mode == 'dilate':
        return cv2.dilate(img, kernel, iterations=1)
    elif mode == 'erode':
        return cv2.erode(img, kernel, iterations=1)
    else:
        return img


img = cv2.imread("C:\\Users\\efegr\\OneDrive\\Belgeler\\PythonProjects\\SignatureAuthentication\\src\\data_prep_pipeline\\01_0104051.png", cv2.IMREAD_GRAYSCALE)


rot = rotate_image(img, angle_deg=20)
trans = translate_image(img, tx=10, ty=-8)
scaled = scale_image(img, scale=0.8)
noisy = add_gaussian_noise(img, sigma=24)
dilated = dilate_erode_image(img, mode='dilate', ksize=2)
eroded = dilate_erode_image(img, mode='erode', ksize=2)

cv2.imwrite("rot.png", rot)
cv2.imwrite("trans.png", trans)
cv2.imwrite("scaled.png", scaled)
cv2.imwrite("noisy.png", noisy)
cv2.imwrite("dilated.png", dilated)
cv2.imwrite("eroded.png", eroded)
