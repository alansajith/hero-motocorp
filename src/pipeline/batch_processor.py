"""Batch processing for multiple images."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from loguru import logger

from .inference import DamageDetectionPipeline


class BatchProcessor:
    """Batch processing for vehicle damage detection."""
    
    def __init__(
        self,
        pipeline: DamageDetectionPipeline,
        max_workers: int = 4
    ):
        """
        Initialize batch processor.
        
        Args:
            pipeline: DamageDetectionPipeline instance
            max_workers: Maximum number of parallel workers
        """
        self.pipeline = pipeline
        self.max_workers = max_workers
        
        logger.info(f"Initialized batch processor with {max_workers} workers")
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str = None,
        image_extensions: List[str] = None,
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing images
            output_dir: Directory to save results (optional)
            image_extensions: List of image file extensions to process
            save_results: Whether to save results to JSON files
            
        Returns:
            List of results for all images
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        
        input_path = Path(input_dir)
        
        # Find all images
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(input_path.glob(f'*{ext}'))
            image_paths.extend(input_path.glob(f'*{ext.upper()}'))
        
        logger.info(f"Found {len(image_paths)} images in {input_dir}")
        
        if len(image_paths) == 0:
            logger.warning("No images found")
            return []
        
        # Process images
        results = []
        
        with tqdm(total=len(image_paths), desc="Processing images") as pbar:
            for image_path in image_paths:
                result = self.pipeline.process_image(str(image_path))
                results.append(result)
                pbar.update(1)
        
        # Save results
        if save_results and output_dir is not None:
            self._save_results(results, output_dir)
        
        logger.info(f"Batch processing complete: {len(results)} images processed")
        
        return results
    
    def process_parallel(
        self,
        image_paths: List[str],
        output_dir: str = None,
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process images in parallel.
        
        Args:
            image_paths: List of image paths
            output_dir: Directory to save results (optional)
            save_results: Whether to save results
            
        Returns:
            List of results
        """
        results = [None] * len(image_paths)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self.pipeline.process_image, path): idx
                for idx, path in enumerate(image_paths)
            }
            
            # Collect results with progress bar
            with tqdm(total=len(image_paths), desc="Processing images") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as e:
                        logger.error(f"Error processing image {image_paths[idx]}: {e}")
                        results[idx] = {
                            'error': str(e),
                            'image_path': image_paths[idx]
                        }
                    pbar.update(1)
        
        # Save results
        if save_results and output_dir is not None:
            self._save_results(results, output_dir)
        
        return results
    
    def _save_results(self, results: List[Dict[str, Any]], output_dir: str):
        """Save results to JSON files."""
        import json
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual results
        for result in results:
            if 'image_path' in result and result['image_path']:
                image_name = Path(result['image_path']).stem
                result_file = output_path / f"{image_name}_result.json"
                
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
        
        # Save summary
        summary_file = output_path / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'total_images': len(results),
                'images_with_damage': sum(1 for r in results if r.get('damage_detected', False)),
                'total_damages': sum(r.get('num_damages', 0) for r in results),
                'results': results
            }, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
