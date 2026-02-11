#!/usr/bin/env python3
"""
LSTV Classification Algorithm
Uses YOLOv11 detections to classify LSTV type
"""

import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
from typing import Dict, List, Tuple, Optional
import cv2

class LSTVClassifier:
    """
    Classify LSTV based on anatomical landmark detections
    
    Classes detected by YOLO:
    0: t12_rib
    1: l5_vertebra
    2: l5_transverse_process
    3: sacrum
    4: l4_vertebra
    """
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        """
        Initialize classifier
        
        Args:
            model_path: Path to trained YOLOv11 weights
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        self.CLASS_T12_RIB = 0
        self.CLASS_L5_VERTEBRA = 1
        self.CLASS_L5_TRANSVERSE = 2
        self.CLASS_SACRUM = 3
        self.CLASS_L4_VERTEBRA = 4
    
    def detect_landmarks(self, image_path: str) -> Dict:
        """
        Run YOLO detection on image
        
        Returns:
            Dictionary of detected landmarks with bounding boxes and confidence
        """
        results = self.model(image_path, conf=self.conf_threshold)[0]
        
        detections = {
            't12_rib': None,
            'l5_vertebra': None,
            'l5_transverse_process': None,
            'sacrum': None,
            'l4_vertebra': None,
        }
        
        if results.boxes is None or len(results.boxes) == 0:
            return detections
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            class_names = ['t12_rib', 'l5_vertebra', 'l5_transverse_process', 'sacrum', 'l4_vertebra']
            class_name = class_names[cls_id]
            
            if detections[class_name] is None or conf > detections[class_name]['confidence']:
                detections[class_name] = {
                    'bbox': xyxy.tolist(),
                    'confidence': conf,
                    'center_y': (xyxy[1] + xyxy[3]) / 2,
                }
        
        return detections
    
    def calculate_l5_sacrum_distance(self, detections: Dict) -> Optional[float]:
        """
        Calculate distance between L5 and sacrum
        Uses vertical distance between bounding box centers
        
        Returns:
            Distance in pixels, or None if either structure not detected
        """
        l5 = detections.get('l5_vertebra')
        sacrum = detections.get('sacrum')
        
        if l5 is None or sacrum is None:
            return None
        
        distance = abs(l5['center_y'] - sacrum['center_y'])
        
        return distance
    
    def classify_lstv(self, detections: Dict, image_height: int) -> Dict:
        """
        Classify LSTV type based on detections
        
        Algorithm:
        1. Check for T12 rib (definitive thoracic marker)
        2. Count vertebrae present
        3. Measure L5-sacrum distance
        4. Check transverse process morphology
        
        Returns:
            Classification result with confidence and reasoning
        """
        
        result = {
            'classification': 'UNCERTAIN',
            'confidence': 0.0,
            'reasoning': [],
            'detections_summary': {},
            'clinical_recommendation': '',
        }
        
        for name, det in detections.items():
            result['detections_summary'][name] = 'DETECTED' if det is not None else 'NOT DETECTED'
        
        has_t12_rib = detections['t12_rib'] is not None
        
        if not has_t12_rib:
            result['reasoning'].append("T12 rib NOT visible - cannot confirm thoracic enumeration")
            result['confidence'] = 0.3
            result['clinical_recommendation'] = "EXTEND FIELD OF VIEW - Include T12 rib for definitive enumeration"
            return result
        
        result['reasoning'].append("T12 rib detected - thoracic enumeration confirmed")
        
        lumbar_count = 0
        if detections['l4_vertebra'] is not None:
            lumbar_count += 1
        if detections['l5_vertebra'] is not None:
            lumbar_count += 1
        
        l5_sacrum_dist = self.calculate_l5_sacrum_distance(detections)
        
        normalized_dist = None
        if l5_sacrum_dist is not None:
            normalized_dist = l5_sacrum_dist / image_height
        
        has_l5_transverse = detections['l5_transverse_process'] is not None
        
        # CLASSIFICATION LOGIC
        
        if detections['l5_vertebra'] is None:
            result['classification'] = 'SACRALIZATION'
            result['confidence'] = 0.7
            result['reasoning'].append("L5 vertebra NOT detected - possible L5-S1 fusion (sacralization)")
            result['reasoning'].append("Appears to be only 4 lumbar vertebrae")
            result['clinical_recommendation'] = "⚠️  LSTV DETECTED - Sacralization suspected. Count vertebrae from T12 down before surgery."
        
        elif normalized_dist is not None and normalized_dist < 0.03:
            result['classification'] = 'SACRALIZATION'
            result['confidence'] = 0.8
            result['reasoning'].append(f"L5 very close to sacrum (distance: {normalized_dist*100:.1f}% of image height)")
            result['reasoning'].append("Suggests L5-S1 fusion (sacralization)")
            result['clinical_recommendation'] = "⚠️  LSTV DETECTED - Sacralization suspected. Verify with whole-spine imaging."
        
        elif normalized_dist is not None and 0.05 <= normalized_dist <= 0.12:
            result['classification'] = 'NORMAL'
            result['confidence'] = 0.85
            result['reasoning'].append(f"Normal L5-S1 distance ({normalized_dist*100:.1f}% of image height)")
            result['reasoning'].append("5 lumbar vertebrae expected from T12")
            result['clinical_recommendation'] = "✓ Normal lumbar anatomy detected"
        
        else:
            result['classification'] = 'UNCERTAIN'
            result['confidence'] = 0.5
            result['reasoning'].append("Ambiguous findings - manual review recommended")
            if normalized_dist:
                result['reasoning'].append(f"L5-S1 distance: {normalized_dist*100:.1f}% of image height")
            result['clinical_recommendation'] = "⚠️  MANUAL REVIEW NEEDED - Borderline LSTV features"
        
        if has_l5_transverse and result['classification'] == 'SACRALIZATION':
            result['reasoning'].append("L5 transverse process detected - confirms fusion morphology")
            result['confidence'] = min(0.95, result['confidence'] + 0.1)
        
        return result
    
    def classify_image(self, image_path: str) -> Dict:
        """
        Full pipeline: detect + classify
        
        Args:
            image_path: Path to sagittal MRI image
            
        Returns:
            Classification result
        """
        img = cv2.imread(str(image_path))
        image_height = img.shape[0]
        
        detections = self.detect_landmarks(image_path)
        
        result = self.classify_lstv(detections, image_height)
        
        result['image_path'] = str(image_path)
        
        return result
    
    def batch_classify(self, image_dir: Path, output_path: Path):
        """
        Classify all images in directory
        
        Args:
            image_dir: Directory containing images
            output_path: Path to save results JSON
        """
        image_files = list(Path(image_dir).glob("*.jpg"))
        
        results = []
        
        print(f"Classifying {len(image_files)} images...")
        
        from tqdm import tqdm
        for img_file in tqdm(image_files):
            result = self.classify_image(str(img_file))
            results.append(result)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        classifications = {}
        for r in results:
            cls = r['classification']
            classifications[cls] = classifications.get(cls, 0) + 1
        
        print("\n" + "="*60)
        print("CLASSIFICATION SUMMARY")
        print("="*60)
        for cls, count in sorted(classifications.items()):
            print(f"{cls:20s}: {count:4d} ({count/len(results)*100:.1f}%)")
        print("="*60)
        
        print(f"\n✓ Results saved to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='LSTV Classification')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to YOLOv11 weights')
    parser.add_argument('--image', type=str, default=None,
                       help='Single image to classify')
    parser.add_argument('--image-dir', type=str, default=None,
                       help='Directory of images to classify')
    parser.add_argument('--output', type=str, default='lstv_classifications.json',
                       help='Output JSON file')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    classifier = LSTVClassifier(args.weights, args.conf)
    
    if args.image:
        result = classifier.classify_image(args.image)
        
        print("\n" + "="*60)
        print("LSTV CLASSIFICATION RESULT")
        print("="*60)
        print(f"Image: {result['image_path']}")
        print(f"Classification: {result['classification']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("\nDetections:")
        for name, status in result['detections_summary'].items():
            print(f"  {name:25s}: {status}")
        print("\nReasoning:")
        for reason in result['reasoning']:
            print(f"  - {reason}")
        print(f"\nRecommendation: {result['clinical_recommendation']}")
        print("="*60)
        
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Result saved to {args.output}")
    
    elif args.image_dir:
        classifier.batch_classify(args.image_dir, args.output)
    
    else:
        print("ERROR: Must provide --image or --image-dir")


if __name__ == "__main__":
    main()
