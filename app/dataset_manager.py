"""
Dataset Management and Training Infrastructure for Magic Card Detection
Implements dataset preparation, augmentation, and training pipelines from the technical guide.
"""

import os
import cv2
import numpy as np
import json
import yaml
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import shutil
from datetime import datetime
import albumentations as A
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetManager:
    """
    Manages dataset creation, organization, and augmentation for card detection training.
    Implements the approach described in section 2 of the technical guide.
    """
    
    def __init__(self, dataset_root: str = "datasets"):
        """
        Initialize dataset manager.
        
        Args:
            dataset_root: Root directory for all datasets
        """
        self.dataset_root = Path(dataset_root)
        self.dataset_root.mkdir(exist_ok=True)
        
        # Standard dataset structure
        self.structure = {
            'train': ['images', 'labels'],
            'valid': ['images', 'labels'],
            'test': ['images', 'labels']
        }
        
        logger.info(f"Dataset manager initialized with root: {self.dataset_root}")
    
    def create_dataset_structure(self, dataset_name: str) -> Path:
        """
        Create standard YOLO dataset structure.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Path to created dataset directory
        """
        dataset_path = self.dataset_root / dataset_name
        
        # Create directory structure
        for split in self.structure:
            for subdir in self.structure[split]:
                (dataset_path / split / subdir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created dataset structure at {dataset_path}")
        return dataset_path
    
    def create_data_yaml(self, dataset_path: Path, class_names: List[str]) -> Path:
        """
        Create data.yaml file for YOLO training.
        
        Args:
            dataset_path: Path to dataset directory
            class_names: List of class names
            
        Returns:
            Path to created data.yaml file
        """
        data_config = {
            'path': str(dataset_path.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(class_names),
            'names': class_names
        }
        
        yaml_path = dataset_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        logger.info(f"Created data.yaml at {yaml_path}")
        return yaml_path
    
    def import_images(self, source_dir: str, dataset_path: Path, 
                     split_ratios: Dict[str, float] = None) -> Dict[str, int]:
        """
        Import images from source directory and split into train/valid/test sets.
        
        Args:
            source_dir: Directory containing source images
            dataset_path: Target dataset directory
            split_ratios: Dictionary with split ratios (default: train=0.7, valid=0.2, test=0.1)
            
        Returns:
            Dictionary with count of images in each split
        """
        if split_ratios is None:
            split_ratios = {'train': 0.7, 'valid': 0.2, 'test': 0.1}
        
        # Validate split ratios
        if abs(sum(split_ratios.values()) - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Get all image files
        source_path = Path(source_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in source_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            logger.warning(f"No images found in {source_dir}")
            return {'train': 0, 'valid': 0, 'test': 0}
        
        # Split images
        train_files, temp_files = train_test_split(
            image_files, 
            train_size=split_ratios['train'], 
            random_state=42
        )
        
        valid_size = split_ratios['valid'] / (split_ratios['valid'] + split_ratios['test'])
        valid_files, test_files = train_test_split(
            temp_files, 
            train_size=valid_size, 
            random_state=42
        )
        
        # Copy files to appropriate directories
        splits = {
            'train': train_files,
            'valid': valid_files,
            'test': test_files
        }
        
        counts = {}
        for split_name, files in splits.items():
            target_dir = dataset_path / split_name / 'images'
            count = 0
            
            for file_path in tqdm(files, desc=f"Copying {split_name} images"):
                target_path = target_dir / file_path.name
                shutil.copy2(file_path, target_path)
                count += 1
            
            counts[split_name] = count
            logger.info(f"Copied {count} images to {split_name} set")
        
        return counts
    
    def create_synthetic_dataset(self, card_images_dir: str, background_images_dir: str,
                               dataset_path: Path, num_samples: int = 1000) -> int:
        """
        Create synthetic dataset by placing card images on various backgrounds.
        Implements the synthetic generation approach from section 2.1.
        
        Args:
            card_images_dir: Directory containing card scan images
            background_images_dir: Directory containing background images
            dataset_path: Target dataset directory
            num_samples: Number of synthetic samples to generate
            
        Returns:
            Number of synthetic samples created
        """
        card_path = Path(card_images_dir)
        bg_path = Path(background_images_dir)
        
        # Get card and background images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        card_images = [f for f in card_path.iterdir() if f.suffix.lower() in image_extensions]
        bg_images = [f for f in bg_path.iterdir() if f.suffix.lower() in image_extensions]
        
        if not card_images:
            logger.error(f"No card images found in {card_images_dir}")
            return 0
        
        if not bg_images:
            logger.error(f"No background images found in {background_images_dir}")
            return 0
        
        # Create synthetic samples
        synthetic_dir = dataset_path / 'synthetic'
        synthetic_dir.mkdir(exist_ok=True)
        (synthetic_dir / 'images').mkdir(exist_ok=True)
        (synthetic_dir / 'labels').mkdir(exist_ok=True)
        
        created_count = 0
        
        for i in tqdm(range(num_samples), desc="Creating synthetic samples"):
            try:
                # Randomly select card and background
                card_img_path = np.random.choice(card_images)
                bg_img_path = np.random.choice(bg_images)
                
                # Load images
                card_img = cv2.imread(str(card_img_path))
                bg_img = cv2.imread(str(bg_img_path))
                
                if card_img is None or bg_img is None:
                    continue
                
                # Create synthetic sample
                synthetic_img, bbox = self._create_synthetic_sample(card_img, bg_img)
                
                if synthetic_img is not None and bbox is not None:
                    # Save synthetic image
                    img_filename = f"synthetic_{i:06d}.jpg"
                    img_path = synthetic_dir / 'images' / img_filename
                    cv2.imwrite(str(img_path), synthetic_img)
                    
                    # Save YOLO format label
                    label_filename = f"synthetic_{i:06d}.txt"
                    label_path = synthetic_dir / 'labels' / label_filename
                    self._save_yolo_label(label_path, bbox, synthetic_img.shape)
                    
                    created_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to create synthetic sample {i}: {e}")
                continue
        
        logger.info(f"Created {created_count} synthetic samples")
        return created_count
    
    def _create_synthetic_sample(self, card_img: np.ndarray, bg_img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
        """
        Create a single synthetic sample by placing card on background.
        
        Args:
            card_img: Card image
            bg_img: Background image
            
        Returns:
            Tuple of (synthetic_image, bbox_normalized) or (None, None) if failed
        """
        try:
            # Resize background to standard size
            target_size = (640, 640)
            bg_resized = cv2.resize(bg_img, target_size)
            
            # Randomly resize card (simulate different distances)
            card_scale = np.random.uniform(0.1, 0.8)
            card_h, card_w = card_img.shape[:2]
            new_card_w = int(card_w * card_scale)
            new_card_h = int(card_h * card_scale)
            card_resized = cv2.resize(card_img, (new_card_w, new_card_h))
            
            # Random rotation
            angle = np.random.uniform(-15, 15)
            center = (new_card_w // 2, new_card_h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            card_rotated = cv2.warpAffine(card_resized, rotation_matrix, (new_card_w, new_card_h))
            
            # Random position on background
            max_x = target_size[0] - new_card_w
            max_y = target_size[1] - new_card_h
            
            if max_x <= 0 or max_y <= 0:
                return None, None
            
            x = np.random.randint(0, max_x)
            y = np.random.randint(0, max_y)
            
            # Place card on background
            result = bg_resized.copy()
            result[y:y+new_card_h, x:x+new_card_w] = card_rotated
            
            # Apply random augmentations
            result = self._apply_augmentations(result)
            
            # Calculate normalized bounding box
            center_x = (x + new_card_w / 2) / target_size[0]
            center_y = (y + new_card_h / 2) / target_size[1]
            width = new_card_w / target_size[0]
            height = new_card_h / target_size[1]
            
            bbox = [center_x, center_y, width, height]
            
            return result, bbox
            
        except Exception as e:
            logger.warning(f"Failed to create synthetic sample: {e}")
            return None, None
    
    def _apply_augmentations(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to image.
        Implements augmentation strategy from section 2.2.
        """
        transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
            A.GaussianBlur(blur_limit=(1, 3), p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        ])
        
        try:
            augmented = transform(image=image)
            return augmented['image']
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}")
            return image
    
    def _save_yolo_label(self, label_path: Path, bbox: List[float], image_shape: Tuple[int, int, int]):
        """
        Save bounding box in YOLO format.
        
        Args:
            label_path: Path to save label file
            bbox: Normalized bounding box [center_x, center_y, width, height]
            image_shape: Shape of the image (height, width, channels)
        """
        # YOLO format: class_id center_x center_y width height
        # For card detection, class_id = 0 (assuming single class)
        with open(label_path, 'w') as f:
            f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
    
    def analyze_dataset(self, dataset_path: Path) -> Dict:
        """
        Analyze dataset and generate statistics.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            'splits': {},
            'total_images': 0,
            'total_labels': 0,
            'class_distribution': {},
            'bbox_statistics': {
                'width_mean': 0, 'width_std': 0,
                'height_mean': 0, 'height_std': 0,
                'area_mean': 0, 'area_std': 0
            }
        }
        
        all_widths, all_heights, all_areas = [], [], []
        class_counts = {}
        
        for split in ['train', 'valid', 'test']:
            split_path = dataset_path / split
            if not split_path.exists():
                continue
            
            images_path = split_path / 'images'
            labels_path = split_path / 'labels'
            
            # Count images and labels
            image_files = list(images_path.glob('*')) if images_path.exists() else []
            label_files = list(labels_path.glob('*.txt')) if labels_path.exists() else []
            
            stats['splits'][split] = {
                'images': len(image_files),
                'labels': len(label_files)
            }
            
            stats['total_images'] += len(image_files)
            stats['total_labels'] += len(label_files)
            
            # Analyze labels
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                width = float(parts[3])
                                height = float(parts[4])
                                area = width * height
                                
                                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                                all_widths.append(width)
                                all_heights.append(height)
                                all_areas.append(area)
                except Exception as e:
                    logger.warning(f"Failed to parse label file {label_file}: {e}")
        
        # Calculate statistics
        if all_widths:
            stats['bbox_statistics']['width_mean'] = np.mean(all_widths)
            stats['bbox_statistics']['width_std'] = np.std(all_widths)
            stats['bbox_statistics']['height_mean'] = np.mean(all_heights)
            stats['bbox_statistics']['height_std'] = np.std(all_heights)
            stats['bbox_statistics']['area_mean'] = np.mean(all_areas)
            stats['bbox_statistics']['area_std'] = np.std(all_areas)
        
        stats['class_distribution'] = class_counts
        
        logger.info(f"Dataset analysis complete: {stats['total_images']} images, {stats['total_labels']} labels")
        return stats
    
    def visualize_dataset(self, dataset_path: Path, split: str = 'train', num_samples: int = 9):
        """
        Visualize dataset samples with annotations.
        
        Args:
            dataset_path: Path to dataset directory
            split: Dataset split to visualize
            num_samples: Number of samples to show
        """
        images_path = dataset_path / split / 'images'
        labels_path = dataset_path / split / 'labels'
        
        if not images_path.exists() or not labels_path.exists():
            logger.error(f"Dataset split {split} not found")
            return
        
        image_files = list(images_path.glob('*'))[:num_samples]
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        for i, img_file in enumerate(image_files):
            if i >= num_samples:
                break
            
            # Load image
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # Load corresponding label
            label_file = labels_path / (img_file.stem + '.txt')
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            center_x = float(parts[1]) * w
                            center_y = float(parts[2]) * h
                            bbox_w = float(parts[3]) * w
                            bbox_h = float(parts[4]) * h
                            
                            # Convert to corner coordinates
                            x1 = int(center_x - bbox_w / 2)
                            y1 = int(center_y - bbox_h / 2)
                            x2 = int(center_x + bbox_w / 2)
                            y2 = int(center_y + bbox_h / 2)
                            
                            # Draw bounding box
                            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            axes[i].imshow(img_rgb)
            axes[i].set_title(f"{img_file.name}")
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(image_files), num_samples):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(dataset_path / f'{split}_samples.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Visualization saved to {dataset_path / f'{split}_samples.png'}")
    
    def export_dataset_info(self, dataset_path: Path) -> Path:
        """
        Export dataset information to JSON file.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Path to exported info file
        """
        stats = self.analyze_dataset(dataset_path)
        
        # Add metadata
        info = {
            'dataset_name': dataset_path.name,
            'created_date': datetime.now().isoformat(),
            'statistics': stats,
            'structure': self.structure
        }
        
        info_path = dataset_path / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Dataset info exported to {info_path}")
        return info_path

class TrainingManager:
    """
    Manages training pipelines for different neural network architectures.
    """
    
    def __init__(self, experiments_dir: str = "experiments"):
        """
        Initialize training manager.
        
        Args:
            experiments_dir: Directory to store training experiments
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)
        logger.info(f"Training manager initialized with experiments dir: {self.experiments_dir}")
    
    def create_experiment(self, experiment_name: str, model_type: str, config: Dict) -> Path:
        """
        Create a new training experiment.
        
        Args:
            experiment_name: Name of the experiment
            model_type: Type of model ('yolo', 'maskrcnn', 'jax')
            config: Training configuration
            
        Returns:
            Path to experiment directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{experiment_name}_{model_type}_{timestamp}"
        exp_path = self.experiments_dir / exp_name
        exp_path.mkdir(exist_ok=True)
        
        # Create experiment structure
        (exp_path / 'checkpoints').mkdir(exist_ok=True)
        (exp_path / 'logs').mkdir(exist_ok=True)
        (exp_path / 'results').mkdir(exist_ok=True)
        
        # Save configuration
        config_path = exp_path / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Created experiment: {exp_name}")
        return exp_path
    
    def train_yolo(self, dataset_path: Path, experiment_path: Path, config: Dict):
        """
        Train YOLOv5 model.
        
        Args:
            dataset_path: Path to dataset
            experiment_path: Path to experiment directory
            config: Training configuration
        """
        try:
            from ultralytics import YOLO
            
            # Initialize model
            model_size = config.get('model_size', 'yolov5s')
            model = YOLO(f'{model_size}.pt')
            
            # Training parameters
            epochs = config.get('epochs', 100)
            batch_size = config.get('batch_size', 16)
            img_size = config.get('img_size', 640)
            
            # Train model
            results = model.train(
                data=str(dataset_path / 'data.yaml'),
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                project=str(experiment_path),
                name='training',
                save_period=10,
                plots=True
            )
            
            # Save final model
            model.save(str(experiment_path / 'final_model.pt'))
            
            logger.info(f"YOLOv5 training completed. Results saved to {experiment_path}")
            return results
            
        except Exception as e:
            logger.error(f"YOLOv5 training failed: {e}")
            return None
    
    def evaluate_model(self, model_path: str, dataset_path: Path, model_type: str) -> Dict:
        """
        Evaluate trained model on test set.
        
        Args:
            model_path: Path to trained model
            dataset_path: Path to dataset
            model_type: Type of model
            
        Returns:
            Evaluation metrics
        """
        try:
            if model_type.lower() == 'yolo':
                from ultralytics import YOLO
                model = YOLO(model_path)
                
                # Run validation
                results = model.val(
                    data=str(dataset_path / 'data.yaml'),
                    split='test'
                )
                
                metrics = {
                    'mAP50': float(results.box.map50),
                    'mAP50-95': float(results.box.map),
                    'precision': float(results.box.mp),
                    'recall': float(results.box.mr),
                    'f1': float(results.box.f1)
                }
                
                logger.info(f"Model evaluation completed: mAP50={metrics['mAP50']:.3f}")
                return metrics
            
            else:
                logger.warning(f"Evaluation not implemented for {model_type}")
                return {}
                
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {}

# Utility functions
def download_sample_data(target_dir: str = "sample_data"):
    """
    Download sample card images for testing.
    This would typically download from Scryfall or other sources.
    """
    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True)
    
    # Placeholder - in real implementation, this would download actual card images
    logger.info(f"Sample data directory created at {target_path}")
    logger.info("Note: Implement actual download logic for card images from Scryfall API")
    
    return target_path