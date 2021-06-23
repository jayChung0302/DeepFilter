# Mosaic
import cv2

def mosaic(img, ratio=0.1):
    small = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def mosaic_area(img, x, y, width, height, ratio=0.1):
    dst = img.copy()
    dst[y:y + height, x:x + width] = mosaic(dst[y:y + height, x:x + width], ratio)
    return dst

if __name__ == '__main__':
    img = cv2.imread('img/Lenna.png')
    dst_area = mosaic_area(img, 220, 230, 150, 150)
    cv2.imwrite('cv-mosaic.jpeg', dst_area)
