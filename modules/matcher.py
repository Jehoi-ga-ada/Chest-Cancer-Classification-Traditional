import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Step 1: Combine SIFT and kaze Features
def extract_combined_features(image, upper_limit=None):
    # SIFT Feature Extraction
    sift = cv2.SIFT_create(nfeatures=100)
    keypoints_sift, descriptors_sift = sift.detectAndCompute(image, None)

    # kaze Feature Extraction
    kaze = cv2.KAZE_create()
    keypoints_kaze, descriptors_kaze = kaze.detectAndCompute(image, None)

    # Handle None descriptors
    if descriptors_sift is None:
        descriptors_sift = np.zeros((1, 128))
    if descriptors_kaze is None:
        descriptors_kaze = np.zeros((1, 32))

    if len(descriptors_sift) > 0:
        norms_sift = np.linalg.norm(descriptors_sift, axis=1, keepdims=True)
        descriptors_sift = descriptors_sift / (norms_sift + 1e-7)  # Avoid division by zero

    # Flatten and concatenate
    combined_features = np.concatenate(
        [descriptors_sift.flatten(), descriptors_kaze.flatten()]
    )

    combined_kp = keypoints_sift, keypoints_kaze

    # Ensure fixed size (upper_limit) by padding or truncating
    if upper_limit == None:
        return combined_features
    elif len(combined_features) < upper_limit:
        combined_features = np.pad(combined_features, (0, upper_limit - len(combined_features)))
    else:
        combined_features = combined_features[:upper_limit]

    return combined_kp, combined_features

# Step 2: Extract Features from Dataset
def extract_features(images, upper_limit=10):
    features = []
    for image in images:
        combined_features = extract_combined_features(image, upper_limit)
        features.append(combined_features)
    return np.array(features)

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2 + 1e-7)

# BFMatcher function
def create_matcher(matcher_type="BF"):
    if matcher_type == "FLANN":
        index_params = dict(algorithm=1, trees=5)  # FLANN parameters
        search_params = dict(checks=50)  # Search parameters
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    elif matcher_type == "BF":
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # BFMatcher with L2 norm
    else:
        raise ValueError("Invalid matcher type. Choose 'FLANN' or 'BF'.")
    return matcher

# Match and visualize matches
def classify_and_plot_matches(query_image, train_images, train_labels, upper_limit=10, matcher_type="BF"):
    """
    Classify the query image by finding the best match among the training images
    based on combined SIFT and KAZE features and visualize matches using BFMatcher or FLANN.
    
    Parameters:
    - query_image: The query image.
    - train_images: List of training images.
    - train_labels: Labels for the training images.
    - upper_limit: Maximum size for the feature vector.
    - matcher_type: Type of matcher ("BF" or "FLANN").
    
    Returns:
    - best_label: The label of the best-matching training image.
    """
    # Extract combined features for the query image
    query_kp, query_features = extract_combined_features(query_image, upper_limit)

    best_label = None
    best_index = -1
    best_similarity = -1  # Start with the worst similarity
    best_train_image = None

    # Compare with each training image
    for i, train_image in tqdm(enumerate(train_images), desc="Matching Query with Train Images"):
        _, train_features = extract_combined_features(train_image, upper_limit)
        similarity = cosine_similarity(query_features, train_features)

        if similarity > best_similarity:
            best_similarity = similarity
            best_label = train_labels[i]
            best_index = i
            best_train_image = train_image

    # Prepare matcher
    matcher = create_matcher(matcher_type)

    # Use both SIFT and KAZE for keypoints and descriptors
    sift = cv2.SIFT_create()
    kaze = cv2.KAZE_create()

    # Extract SIFT and KAZE keypoints and descriptors for the query image
    query_kp_sift, query_desc_sift = sift.detectAndCompute(query_image, None)
    query_kp_kaze, query_desc_kaze = kaze.detectAndCompute(query_image, None)

    # Extract SIFT and KAZE keypoints and descriptors for the best matching training image
    train_kp_sift, train_desc_sift = sift.detectAndCompute(best_train_image, None)
    train_kp_kaze, train_desc_kaze = kaze.detectAndCompute(best_train_image, None)

    # Ensure descriptors are not None
    if query_desc_sift is None:
        query_desc_sift = np.zeros((1, 128))  # SIFT descriptor size
    if query_desc_kaze is None:
        query_desc_kaze = np.zeros((1, 64))  # KAZE descriptor size

    if train_desc_sift is None:
        train_desc_sift = np.zeros((1, 128))
    if train_desc_kaze is None:
        train_desc_kaze = np.zeros((1, 64))

    # Match the size of descriptors by padding the smaller one
    if len(query_desc_sift) < len(query_desc_kaze):
        query_desc_sift = np.pad(query_desc_sift, ((0, len(query_desc_kaze) - len(query_desc_sift)), (0, 0)), 'constant')
    elif len(query_desc_kaze) < len(query_desc_sift):
        query_desc_kaze = np.pad(query_desc_kaze, ((0, len(query_desc_sift) - len(query_desc_kaze)), (0, 0)), 'constant')

    if len(train_desc_sift) < len(train_desc_kaze):
        train_desc_sift = np.pad(train_desc_sift, ((0, len(train_desc_kaze) - len(train_desc_sift)), (0, 0)), 'constant')
    elif len(train_desc_kaze) < len(train_desc_sift):
        train_desc_kaze = np.pad(train_desc_kaze, ((0, len(train_desc_sift) - len(train_desc_kaze)), (0, 0)), 'constant')

    # Combine descriptors from both SIFT and KAZE
    query_desc_combined = np.hstack((query_desc_sift, query_desc_kaze))
    train_desc_combined = np.hstack((train_desc_sift, train_desc_kaze))

    # Match descriptors using BFMatcher or FLANN
    matches = matcher.knnMatch(query_desc_combined, train_desc_combined, k=2)

    # Apply ratio test (Lowe's ratio test)
    good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]

    # Draw matches
    match_img = cv2.drawMatchesKnn(
        query_image, query_kp_sift + query_kp_kaze,  # Combined keypoints (SIFT + KAZE)
        best_train_image, train_kp_sift + train_kp_kaze,  # Combined keypoints (SIFT + KAZE)
        [[m] for m in good_matches],
        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Visualize the query image and best-matching training image along with matches
    plt.figure(figsize=(15, 10))
    plt.title(f"Query Image vs Best Match (Label: {best_label})")
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.axis(False)

    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(query_image)
    plt.axis(False)
    plt.subplot(1, 2, 2)
    plt.imshow(best_train_image)
    plt.axis(False)
    plt.show()

    return best_label