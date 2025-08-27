# Codes/image_enhancer.py
import cv2, numpy as np
from pathlib import Path

def enhance_image(image):
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(image, -1, kernel)
    if len(sharp.shape) == 3 and sharp.shape[2] == 3:
        ycrcb = cv2.cvtColor(sharp, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    elif len(sharp.shape) == 2:
        return cv2.equalizeHist(sharp)
    return sharp


def process_input_images(input_dir: Path = Path("input"), output_dir: Path = Path("temp_upload")):
    output_dir.mkdir(parents=True, exist_ok=True)
    imgs = []
    for ext in ("*.tif","*.png","*.jpg","*.jpeg"):
        imgs.extend(list(input_dir.glob(ext)))
    for p in imgs:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None: continue
        enh = enhance_image(img)
        out = output_dir / p.name
        cv2.imwrite(str(out), enh)