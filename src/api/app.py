"""FastAPI application for vehicle damage detection."""

import cv2
import numpy as np
import time
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
from typing import Optional
from loguru import logger
import sys

from .schemas import DetectionResponse, HealthResponse, ErrorResponse
from ..pipeline.inference import DamageDetectionPipeline
from ..visualization.report_generator import ReportGenerator
from ..utils.config_loader import get_config

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/api.log", rotation="500 MB", level="DEBUG")

# Initialize FastAPI app
app = FastAPI(
    title="Vehicle Damage Detection API",
    description="AI-powered vehicle damage detection, classification, and assessment",
    version="1.0.0"
)

# Load configuration
config = get_config()

# Add CORS middleware
if config.get('api.cors_enabled', True):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get('api.cors_origins', ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Global pipeline instance (lazy loading)
pipeline: Optional[DamageDetectionPipeline] = None
report_generator: Optional[ReportGenerator] = None


def get_pipeline() -> DamageDetectionPipeline:
    """Get or create pipeline instance."""
    global pipeline
    if pipeline is None:
        logger.info("Initializing damage detection pipeline...")
        models_dir = Path(config.models_dir)
        
        # Look for trained models
        detection_model = models_dir / "detection" / "best.pt"
        segmentation_model = models_dir / "segmentation" / "best.pt"
        classification_model = models_dir / "classification" / "best.pt"
        
        pipeline = DamageDetectionPipeline(
            detection_model_path=str(detection_model) if detection_model.exists() else None,
            segmentation_model_path=str(segmentation_model) if segmentation_model.exists() else None,
            classification_model_path=str(classification_model) if classification_model.exists() else None,
        )
        logger.info("Pipeline initialized successfully")
    
    return pipeline


def get_report_generator() -> ReportGenerator:
    """Get or create report generator."""
    global report_generator
    if report_generator is None:
        report_generator = ReportGenerator()
    return report_generator


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Starting Vehicle Damage Detection API...")
    logger.info(f"Using device: {config.device}")
    
    # Pre-load models
    try:
        get_pipeline()
        get_report_generator()
        logger.info("API ready to accept requests")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "name": "Vehicle Damage Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "detect": "/detect",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        pipe = get_pipeline()
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            device=config.device,
            models_loaded={
                "detection": pipe.detector is not None,
                "segmentation": pipe.segmentation_model is not None,
                "classification": pipe.classifier is not None,
                "part_identification": pipe.part_identifier is not None,
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect", response_model=DetectionResponse)
async def detect_damage(
    file: UploadFile = File(..., description="Image file to analyze"),
    generate_report: bool = False,
    background_tasks: BackgroundTasks = None
):
    """
    Detect and assess vehicle damage in an uploaded image.
    
    Args:
        file: Image file (JPG, PNG, etc.)
        generate_report: Whether to generate a comprehensive report
        
    Returns:
        Damage detection results
    """
    # Validate file type
    allowed_extensions = config.get('api.allowed_extensions', ['jpg', 'jpeg', 'png'])
    file_ext = file.filename.split('.')[-1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Read image
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        logger.info(f"Processing image: {file.filename} ({image.shape})")
    except Exception as e:
        logger.error(f"Failed to read image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    # Process image
    try:
        start_time = time.time()
        
        pipe = get_pipeline()
        results = pipe.process_image(image)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        results['processing_time_ms'] = processing_time
        
        logger.info(f"Processing completed in {processing_time:.2f}ms")
        
        # Generate report if requested
        if generate_report and results.get('damage_detected', False):
            if background_tasks:
                background_tasks.add_task(
                    get_report_generator().generate_report,
                    image,
                    results,
                    config.outputs_dir,
                    file.filename.split('.')[0]
                )
        
        return DetectionResponse(**results)
    
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/batch")
async def detect_batch(
    files: list[UploadFile] = File(..., description="Multiple image files"),
):
    """
    Process multiple images in batch.
    
    Args:
        files: List of image files
        
    Returns:
        List of detection results
    """
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    max_batch_size = 10
    if len(files) > max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum of {max_batch_size}"
        )
    
    results = []
    pipe = get_pipeline()
    
    for file in files:
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            result = pipe.process_image(image)
            result['filename'] = file.filename
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            results.append({
                'filename': file.filename,
                'error': str(e),
                'damage_detected': False
            })
    
    return JSONResponse(content={'results': results, 'total': len(results)})


@app.get("/models/info")
async def get_models_info():
    """Get information about loaded models."""
    pipe = get_pipeline()
    
    return {
        "detection_model": {
            "loaded": pipe.detector is not None,
            "type": "YOLOv8 Segmentation",
            "classes": pipe.detector.num_classes if pipe.detector else 0,
        },
        "segmentation_model": {
            "loaded": pipe.segmentation_model is not None,
            "type": "U-Net",
        },
        "classification_model": {
            "loaded": pipe.classifier is not None,
            "type": "EfficientNet",
            "classes": pipe.classifier.num_classes if pipe.classifier else 0,
        },
        "part_identification": {
            "loaded": pipe.part_identifier is not None,
            "enabled": pipe.use_part_identification,
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=config.get('api.host', '0.0.0.0'),
        port=config.get('api.port', 8000),
        reload=config.get('api.reload', False)
    )
