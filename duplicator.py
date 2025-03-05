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

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

class ROIFeatureExtractor:
    def __init__(self, yolo_weights, resnet_device='cpu'):  # Default to 'cpu'
        # YOLO model initialization
        self.yolo_model = YOLO(yolo_weights)
        
        # ResNet model setup
        self.resnet_device = resnet_device
        self.resnet_model = models.resnet50(pretrained=True)
        self.resnet_model = nn.Sequential(*list(self.resnet_model.children())[:-1])
        self.resnet_model.eval()
        self.resnet_model.to(self.resnet_device)  # Ensure it's using the correct device

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_roi_features(self, image_path, conf=0.5, iou=0.5, classes=None):
        """
        提取圖片中特定ROI的特徵向量
        
        :param image_path: 圖片路徑
        :param conf: 偵測置信度閾值
        :param iou: IoU閾值
        :param classes: 要偵測的類別列表，None表示全部類別
        :return: ROI特徵向量列表
        """
        # 讀取圖片
        image = cv2.imread(image_path)
        
        # YOLO偵測
        results = self.yolo_model(image, conf=conf, iou=iou, classes=classes)
        
        roi_features = []
        for result in results:
            # 獲取偵測框
            boxes = result.boxes
            
            for box in boxes:
                # 獲取邊界框座標
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                
                # 裁剪ROI
                roi = image[y1:y2, x1:x2]
                
                # 如果ROI為空，跳過
                if roi.size == 0:
                    continue
                
                # PIL轉換和預處理
                roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                roi_tensor = self.transform(roi_pil)
                roi_tensor = roi_tensor.unsqueeze(0).to(self.resnet_device)
                
                # 提取特徵
                with torch.no_grad():
                    features = self.resnet_model(roi_tensor)
                    features = features.view(features.size(0), -1)
                
                roi_features.append({
                    'features': features.cpu().numpy().astype(np.float32)[0],
                    'bbox': (x1, y1, x2, y2),
                    'class': self.yolo_model.names[int(box.cls[0])],
                    'image_path': image_path  # Add image path to ROI info
                })
        
        return roi_features

def compute_roi_similarity_matrix(roi_features_list):
    """
    計算ROI特徵間的餘弦相似度矩陣
    
    :param roi_features_list: 每張圖的ROI特徵列表
    :return: 相似度矩陣和ROI配對資訊
    """
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
    
    # 計算相似度矩陣
    all_features_array = np.array(all_roi_features)
    norms = np.linalg.norm(all_features_array, axis=1, keepdims=True)
    normalized = all_features_array / (norms + 1e-10)
    similarity_matrix = np.dot(normalized, normalized.T)
    
    return similarity_matrix, roi_mapping

def find_similar_rois(similarity_matrix, roi_mapping, threshold=0.95):
    """
    找出相似的ROI對
    
    :param similarity_matrix: ROI相似度矩陣
    :param roi_mapping: ROI對應資訊
    :param threshold: 相似度閾值
    :return: 相似ROI對列表
    """
    similar_rois = []
    N = similarity_matrix.shape[0]
    
    for i in range(N):
        for j in range(i+1, N):
            # 排除同一張圖的ROI
            if roi_mapping[i]['img_index'] == roi_mapping[j]['img_index']:
                continue
            
            sim_score = similarity_matrix[i, j]
            if sim_score >= threshold:
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

def main():
    # 參數設定
    folder_list = [
        r"/Users/vier/Downloads/cam26_20231004_0945/images"
    ]
    yolo_weights = "/Users/vier/Downloads/ppe_WithMask_WithoutNoHelmet_250124_best 2.pt"
    threshold = 0.95
    classes = None  # 指定要偵測的類別 (None 表示全部)
    
    # 初始化 ROI 特徵提取器
    extractor = ROIFeatureExtractor(yolo_weights)
    
    # 取得圖片路徑
    image_paths = []
    for folder in folder_list:
        for root, dirs, files in os.walk(folder):
            image_paths.extend([os.path.join(root, f) for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # 提取所有圖片的ROI特徵
    roi_features_list = []
    for path in tqdm(image_paths, desc="Extracting ROI features"):
        roi_features = extractor.extract_roi_features(path, classes=classes)
        roi_features_list.append(roi_features)
    
    # 計算ROI相似度矩陣
    similarity_matrix, roi_mapping = compute_roi_similarity_matrix(roi_features_list)
    
    # 找出相似的ROI
    similar_rois = find_similar_rois(similarity_matrix, roi_mapping, threshold)
    
    # 輸出結果到 JSON 文件
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

    # 設定結果的 JSON 文件路徑
    result_dir = "/Users/vier/Downloads/test"

    json_path = os.path.join(result_dir, "report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(similar_rois_output, f, indent=4, ensure_ascii=False)

    print(f"\nResult JSON: {json_path}")
    print(f"Done. Output folder: {result_dir}")

if __name__ == "__main__":
    main()
