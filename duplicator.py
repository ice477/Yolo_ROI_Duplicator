import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import json
import random
from datetime import datetime
import ultralytics
from ultralytics import YOLO
from collections import Counter

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

class ROIFeatureExtractor:
    def __init__(self, yolo_weights, resnet_device='cpu'):
        self.yolo_model = YOLO(yolo_weights)
        
        self.resnet_device = resnet_device
        self.resnet_model = models.resnet50(pretrained=True)
        self.resnet_model = nn.Sequential(*list(self.resnet_model.children())[:-1])
        self.resnet_model.eval()
        self.resnet_model.to(self.resnet_device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_roi_features(self, image_path, conf=0.5, iou=0.5, classes=None):
        image = cv2.imread(image_path)
        
        if classes is not None:
            class_indices = [self.yolo_model.names.index(class_name) for class_name in classes if class_name in self.yolo_model.names]
        else:
            class_indices = None
        
        results = self.yolo_model(image, conf=conf, iou=iou, classes=class_indices)
        
        roi_features = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                roi = image[y1:y2, x1:x2]
                
                if roi.size == 0:
                    continue
                
                roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                roi_tensor = self.transform(roi_pil)
                roi_tensor = roi_tensor.unsqueeze(0).to(self.resnet_device)
                
                with torch.no_grad():
                    features = self.resnet_model(roi_tensor)
                    features = features.view(features.size(0), -1)
                
                class_name = self.yolo_model.names[int(box.cls[0])]
                if classes is None or class_name in classes:
                    roi_features.append({
                        'features': features.cpu().numpy().astype(np.float32)[0],
                        'bbox': (x1, y1, x2, y2),
                        'class': class_name,
                        'image_path': image_path
                    })
        
        return roi_features

def compute_roi_similarity_matrix(roi_features_list):
    all_roi_features = []
    roi_mapping = []
    
    for img_index, roi_features in enumerate(roi_features_list):
        for roi_info in roi_features:
            all_roi_features.append(roi_info['features'])
            roi_mapping.append({
                'img_index': img_index,
                'bbox': roi_info['bbox'],
                'class': roi_info['class'],
                'image_path': roi_info['image_path']
            })
    
    if not all_roi_features:  # Avoid empty similarity calculation
        return np.array([]), roi_mapping

    all_features_array = np.array(all_roi_features)
    norms = np.linalg.norm(all_features_array, axis=1, keepdims=True)
    normalized = all_features_array / (norms + 1e-10)
    similarity_matrix = np.dot(normalized, normalized.T)
    
    return similarity_matrix, roi_mapping

def find_similar_rois(similarity_matrix, roi_mapping, threshold=0.95):
    similar_rois = []
    N = similarity_matrix.shape[0]
    
    for i in range(N):
        for j in range(i+1, N):
            if roi_mapping[i]['img_index'] == roi_mapping[j]['img_index']:
                continue
            
            sim_score = similarity_matrix[i, j]
            if sim_score >= threshold:
                # Only compare similar rois if both are in the specified classes
                if roi_mapping[i]['class'] == roi_mapping[j]['class']:
                    similar_rois.append({
                        'image1': roi_mapping[i]['image_path'],
                        'image2': roi_mapping[j]['image_path'],
                        'roi1': roi_mapping[i]['bbox'],
                        'roi2': roi_mapping[j]['bbox'],
                        'class1': roi_mapping[i]['class'],
                        'class2': roi_mapping[j]['class'],
                        'similarity': float(sim_score)
                    })
    
    return similar_rois

def analyze_results(similar_rois, roi_mapping, thereshold):
    total_rois = len(roi_mapping)
    total_images = len(set([roi['image_path'] for roi in roi_mapping]))

    # 計算重複圖像數量
    duplicate_images = set()
    for roi in similar_rois:
        duplicate_images.add(roi['image1'])
        duplicate_images.add(roi['image2'])
    
    # 只計算重複的圖在所有圖中的比率
    duplicate_count = len(duplicate_images)
    duplicate_rate = (duplicate_count / total_images) * 100 if total_images > 0 else 0
    
    roi_bboxes = [str(roi['roi1']) for roi in similar_rois]
    duplicate_bboxes = Counter(roi_bboxes)
    most_frequent_duplicates = duplicate_bboxes.most_common(10)
    
    highly_duplicated_rois = [roi for roi in similar_rois if roi_bboxes.count(str(roi['roi1'])) > 3]
    
    return {
        'current threshold': thereshold,
        'total_images': total_images,
        'total_rois': total_rois,
        'duplicate_count': duplicate_count,
        'duplicate_rate': duplicate_rate,
        'most_frequent_duplicates': most_frequent_duplicates,
        'highly_duplicated_rois': highly_duplicated_rois
    }

def average_similarity_by_classes(similar_rois):
    class_pairs_similarity = {}

    for roi in similar_rois:
        # Convert class pair tuple into a string like 'class1_class2'
        class_pair = "_".join(sorted([roi['class1'], roi['class2']]))
        if class_pair not in class_pairs_similarity:
            class_pairs_similarity[class_pair] = []
        class_pairs_similarity[class_pair].append(roi['similarity'])
    
    for class_pair, similarities in class_pairs_similarity.items():
        avg_similarity = np.mean(similarities)
        class_pairs_similarity[class_pair] = avg_similarity
    
    return class_pairs_similarity

def main():
    # Define the folder list, YOLO weights, and other parameters
    folder_list = [r"/path/to/your/images"]
    yolo_weights = "/path/to/your/model.pt"
    threshold = 0.975
    classes = None # None means detect all classes
    
    # Initialize the extractor with YOLO weights
    extractor = ROIFeatureExtractor(yolo_weights)
    
    # Rest of your code for processing images
    image_paths = []
    for folder in folder_list:
        for root, dirs, files in os.walk(folder):
            image_paths.extend([os.path.join(root, f) for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    roi_features_list = []
    for path in tqdm(image_paths, desc="Extracting ROI features"):
        roi_features = extractor.extract_roi_features(path, classes=classes)  # Use the extractor here
        roi_features_list.append(roi_features)
 
    similarity_matrix, roi_mapping = compute_roi_similarity_matrix(roi_features_list)
    if similarity_matrix.size == 0:  # Skip further steps if no features for the specified classes
        print("No features found for the specified classes. Exiting...")
        return
    
    similar_rois = find_similar_rois(similarity_matrix, roi_mapping, threshold)
    
    analysis_results = analyze_results(similar_rois, roi_mapping, threshold)
    class_similarity = average_similarity_by_classes(similar_rois)
    
    similar_rois_output = []
    for similar_roi in similar_rois:
        output = {
            "image1": similar_roi['image1'],
            "image2": similar_roi['image2'],
            "roi1": similar_roi['roi1'],
            "roi2": similar_roi['roi2'],
            "class1": similar_roi['class1'],
            "class2": similar_roi['class2'],
            "similarity": similar_roi['similarity']
        }
        similar_rois_output.append(output)
    
    result_dir = "/path/to/your/results"
    os.makedirs(result_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(result_dir, timestamp_str)
    os.makedirs(result_dir, exist_ok=True)

    json_path = os.path.join(result_dir, "report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "analysis": analysis_results,
            "similar_rois": similar_rois_output
        }, f, indent=4, ensure_ascii=False)
    
    print(f"\nResult JSON: {json_path}")
    print(f"Done. Output folder: {result_dir}")

if __name__ == "__main__":
    main()