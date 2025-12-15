import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.morphology import skeletonize


def segment_and_identify_roots(mask, n_sections=5, y_threshold_start=0.1, y_threshold_end=0.2):
    """
    Divide image into X sections and identify the largest root component in each.
    Sections without valid roots are skipped (not included in output).
    
    Args:
        mask: Binary mask
        n_sections: Number of vertical sections to divide image into
        y_threshold_start: Start of Y threshold as ratio (0-1)
        y_threshold_end: End of Y threshold as ratio (0-1)
    
    Returns:
        section_roots: Dict mapping section index to component label (only sections with roots)
        stats: Component statistics from cv2.connectedComponentsWithStats
        labels: Label image from cv2.connectedComponentsWithStats
    """
    h, w = mask.shape
    
    # Calculate Y threshold in pixels
    y_start = int(h * y_threshold_start)
    y_end = int(h * y_threshold_end)
    
    # Get connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Calculate section boundaries
    section_width = w / n_sections
    
    section_roots = {}
    
    for section_idx in range(n_sections):
        # Calculate X boundaries for this section
        x_start = section_idx * section_width
        x_end = (section_idx + 1) * section_width
        
        candidates = []
        
        # Check each component (skip background label 0)
        for label_idx in range(1, num_labels):
            left = stats[label_idx, cv2.CC_STAT_LEFT]
            top = stats[label_idx, cv2.CC_STAT_TOP]
            width_comp = stats[label_idx, cv2.CC_STAT_WIDTH]
            area = stats[label_idx, cv2.CC_STAT_AREA]
            
            # Calculate center X position
            center_x = left + width_comp / 2
            
            # Check if component is in this X section and Y threshold
            if x_start <= center_x < x_end and y_start <= top <= y_end:
                candidates.append((label_idx, area))
        
        # Select largest component in this section (only if candidates exist)
        if candidates:
            largest_label = max(candidates, key=lambda x: x[1])[0]
            section_roots[section_idx] = largest_label
        # If no candidates, section_idx is not added to dict (skipped)
    
    return section_roots, stats, labels


def crop_to_root(mask, labels, component_label, stats, padding=10):
    """
    Crop image to specific root component with padding.
    
    Args:
        mask: Original binary mask
        labels: Label image from connectedComponentsWithStats
        component_label: Label of the component to keep
        stats: Statistics from connectedComponentsWithStats
        padding: Padding around bounding box
    
    Returns:
        cropped_mask: Cropped mask containing only the specified component
        crop_info: Dict with crop coordinates
    """
    # Get bounding box
    left = stats[component_label, cv2.CC_STAT_LEFT]
    top = stats[component_label, cv2.CC_STAT_TOP]
    width = stats[component_label, cv2.CC_STAT_WIDTH]
    height = stats[component_label, cv2.CC_STAT_HEIGHT]
    
    # Add padding
    h, w = mask.shape
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(w, left + width + 2 * padding)
    bottom = min(h, top + height + 2 * padding)
    
    # Create mask with only this component
    component_mask = (labels == component_label).astype(np.uint8)
    
    # Crop
    cropped_mask = component_mask[top:bottom, left:right]
    
    crop_info = {
        'left': left,
        'top': top,
        'right': right,
        'bottom': bottom,
        'width': right - left,
        'height': bottom - top
    }
    
    return cropped_mask, crop_info



def get_bottom(predicted_mask, preprocess_info):
    
    # Apply multi-directional closing if needed
    whole_plant_uint8 = (predicted_mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 55))  # TALL
    whole_plant_uint8 = cv2.morphologyEx(
        whole_plant_uint8, cv2.MORPH_CLOSE, kernel, iterations=1
    )
    
    # Load original image for display
    
    
    
    # Identify roots in each section
    section_roots, stats, labels = segment_and_identify_roots(
        whole_plant_uint8,
        n_sections=5,
        y_threshold_start=0.1,  # Top 10% to 20% of image
        y_threshold_end=0.3
    )
    
    # Filter labels to keep only the identified roots
    keep_list = list(section_roots.values())
    labels_filtered = labels.copy()
    labels_filtered[~np.isin(labels_filtered, keep_list)] = 0

    prediction = predicted_mask.reshape(labels.shape)

    # Dictionary to store the absolute bottom Y coordinate
    root_bottoms = {}
    
    # Intersect the closed components with the original prediction 
    # to ensure we are measuring the actual predicted pixels, not the 'closed' morphology
    mask_for_label = labels_filtered * prediction

    for section_idx, label_id in section_roots.items():
        # Crop to the specific root component
        cropped_root, crop_info = crop_to_root(mask_for_label, mask_for_label, label_id, stats, padding=10)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        closed = cv2.morphologyEx(cropped_root, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
        
        
        skeleton = skeletonize(closed)

        points = np.argwhere(skeleton > 0)

        if len(points) == 0:
            lowest_point_local = None
        else:
            lowest_point_local = points[np.argmax(points[:, 0])]
            local_y, local_x = lowest_point_local
            
            mask_y = local_y + crop_info['top']
            mask_x = local_x + crop_info['left']

            abs_y = mask_y + preprocess_info.crop_y
            abs_x = mask_x + preprocess_info.crop_x
            # Store as (x, y) for plotting, or (y, x) depending on your preference. 
            # Standard for plotting (plt.scatter) is (x, y). Standard for array indexing is (y, x).
            # Here I return (x, y) which is easier for plotting later.
            root_bottoms[section_idx] = (mask_x, mask_y)
            
            
            
            
        
        
    
    

    

    return root_bottoms