import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops

def preprocess_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    return edges

def load_images(annotation): 
    images = []
    for path in annotation['path'].values:
        image = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        image = preprocess_image(image)
        images.append(image)
    return np.array(images)

def extract_combined_features(images, max_descriptors=100):
    sift = cv2.SIFT_create()
    kaze = cv2.KAZE_create()
    combined_features = []
    
    for image in tqdm(images, desc="Extracting SIFT + KAZE features"):
        # Sobel Edge Detection
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=11)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=11)
        sobel_edges = cv2.magnitude(sobel_x, sobel_y)
        
        # Texture Features (Haralick features)
        sobel_edges_uint8 = np.uint8(sobel_edges / np.max(sobel_edges) * 255)
        glcm = graycomatrix(sobel_edges_uint8, distances=[1], angles=[0], levels=256)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        
        # Histogram of Sobel edges
        sobel_hist, _ = np.histogram(sobel_edges.ravel(), bins=256, range=(0, 255), density=True)
        sobel_hist = sobel_hist / np.sum(sobel_hist)  # Normalized histogram
        
        # SIFT Features
        _, sift_descriptors = sift.detectAndCompute(image, None)
        if sift_descriptors is None:
            sift_descriptors = np.zeros((0, 128), dtype=np.float32)
        elif sift_descriptors.shape[0] > max_descriptors:
            sift_descriptors = sift_descriptors[:max_descriptors]
        else:
            sift_descriptors = np.pad(
                sift_descriptors, 
                ((0, max_descriptors - sift_descriptors.shape[0]), (0, 0)), 
                'constant'
            ).astype(np.float32)

        # kaze Features
        _, kaze_descriptors = kaze.detectAndCompute(image, None)
        if kaze_descriptors is None:
            kaze_descriptors = np.zeros((0, 32), dtype=np.float32)
        elif kaze_descriptors.shape[0] > max_descriptors:
            kaze_descriptors = kaze_descriptors[:max_descriptors]
        else:
            kaze_descriptors = np.pad(
                kaze_descriptors, 
                ((0, max_descriptors - kaze_descriptors.shape[0]), (0, 0)), 
                'constant'
            ).astype(np.float32)

        # Concatenate histogram, Sobel, texture features, and SIFT + kaze descriptors
        combined_feature = np.concatenate([sobel_hist, [contrast, correlation, energy, homogeneity], sift_descriptors.flatten(), kaze_descriptors.flatten()])
        combined_features.append(combined_feature)
    
    return combined_features