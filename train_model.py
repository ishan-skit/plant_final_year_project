import os
import json
import sys
import logging
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np
import gc
from PIL import Image

# Configure logging for Render (removed emojis for Windows compatibility)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log') if os.access('.', os.W_OK) else logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# CRITICAL: CPU-only configuration optimized for Render
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'  # Conservative for Render's shared environment
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

# Render-specific memory optimization
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Configure TensorFlow for Render deployment
try:
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    # Enable memory optimization
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logger.info("[SUCCESS] TensorFlow configured successfully for Render deployment")
except Exception as e:
    logger.warning(f"[WARNING] TensorFlow configuration warning: {e}")

logger.info("[START] Starting model training with CPU-only configuration for Render...")
logger.info(f"TensorFlow version: {tf.__version__}")
logger.info(f"Python version: {sys.version}")

# CRITICAL: Configuration matching app.py
DATA_DIR = os.environ.get('DATA_DIR', 'PlantVillage')  # Allow environment override
IMG_SIZE = (128, 128)  # MUST match app.py exactly
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '8'))  # Smaller for Render's memory limits
INPUT_SHAPE = (128, 128, 3)
EPOCHS = int(os.environ.get('EPOCHS', '20'))  # Allow environment override

logger.info(f"[CONFIG] Data directory: {DATA_DIR}")
logger.info(f"[CONFIG] Image size: {IMG_SIZE}")
logger.info(f"[CONFIG] Batch size: {BATCH_SIZE}")
logger.info(f"[CONFIG] Epochs: {EPOCHS}")

# Memory optimization for Render
tf.keras.backend.clear_session()
gc.collect()

def check_environment():
    """Check if running environment is suitable for training"""
    try:
        # Check available memory (rough estimate)
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        logger.info(f"[MEMORY] Available memory: {memory_gb:.1f} GB")
        
        if memory_gb < 1.0:
            logger.warning("[WARNING] Low memory detected. Reducing batch size...")
            return max(4, BATCH_SIZE // 2)
    except ImportError:
        logger.info("[INFO] Memory check skipped (psutil not available)")
    
    return BATCH_SIZE

def verify_data_directory():
    """Verify data directory exists and has proper structure"""
    if not os.path.exists(DATA_DIR):
        logger.error(f"[ERROR] Data directory '{DATA_DIR}' not found!")
        logger.error("Please ensure your dataset is uploaded to Render and accessible")
        
        # Try alternative locations
        possible_dirs = ['./PlantVillage', '../PlantVillage', '/opt/render/project/src/PlantVillage']
        for alt_dir in possible_dirs:
            if os.path.exists(alt_dir):
                logger.info(f"[FOUND] Alternative data directory: {alt_dir}")
                return alt_dir
        
        raise FileNotFoundError(f"Dataset not found in any expected location")
    
    # Verify structure
    subdirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if len(subdirs) == 0:
        raise ValueError(f"No class directories found in {DATA_DIR}")
    
    logger.info(f"[SUCCESS] Found {len(subdirs)} class directories in dataset")
    return DATA_DIR

def create_optimized_generators(data_dir, batch_size):
    """Create data generators optimized for Render deployment"""
    
    # Conservative augmentation for stability
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # CRITICAL: Same normalization as app.py
        validation_split=0.2,
        rotation_range=10,  # Reduced for stability on Render
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        zoom_range=0.05,
        fill_mode='nearest'
    )

    # Validation generator - EXACT same preprocessing as app.py
    val_datagen = ImageDataGenerator(
        rescale=1./255,  # CRITICAL: Exact same as app.py
        validation_split=0.2
    )

    try:
        logger.info("[LOADING] Loading training data...")
        train_gen = train_datagen.flow_from_directory(
            data_dir,
            target_size=IMG_SIZE,
            batch_size=batch_size,
            class_mode='sparse',
            subset='training',
            shuffle=True,
            seed=42,
            interpolation='lanczos'
        )

        logger.info("[LOADING] Loading validation data...")
        val_gen = val_datagen.flow_from_directory(
            data_dir,
            target_size=IMG_SIZE,
            batch_size=batch_size,
            class_mode='sparse',
            subset='validation',
            shuffle=False,
            seed=42,
            interpolation='lanczos'
        )

        return train_gen, val_gen

    except Exception as e:
        logger.error(f"[ERROR] Error loading data: {e}")
        raise

def build_lightweight_model(num_classes):
    """Build lightweight CNN optimized for Render deployment with explicit Input layer"""
    
    logger.info("[BUILD] Building lightweight CNN model for Render...")
    
    # Ultra-lightweight model for Render's resource constraints with explicit Input layer
    model = Sequential([
        # FIXED: Explicit Input layer instead of input_shape in Conv2D
        Input(shape=INPUT_SHAPE),
        
        # First Block - Minimal filters
        Conv2D(16, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.1),
        
        # Second Block
        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.1),
        
        # Third Block
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.2),
        
        # Fully Connected Layers - Optimized for speed
        Flatten(),
        Dense(128, activation='relu'),  # Reduced for faster inference
        Dropout(0.4),
        Dense(64, activation='relu'),   # Additional reduction
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    # CRITICAL: Exact same compilation as app.py
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        run_eagerly=False
    )

    return model

def train_model_with_checkpoints(model, train_gen, val_gen):
    """Train model with proper checkpointing for Render"""
    
    # Ensure model directory exists
    os.makedirs('model', exist_ok=True)
    
    # Render-optimized callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,  # Reduced patience for faster deployment
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # More aggressive reduction
            patience=3,
            min_lr=0.00001,
            verbose=1
        ),
        ModelCheckpoint(
            'model/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max',
            save_weights_only=False
        )
    ]

    try:
        # Clear session before training
        tf.keras.backend.clear_session()
        gc.collect()
        
        logger.info("[TRAINING] Starting training...")
        
        # FIXED: Removed incompatible arguments (workers, use_multiprocessing, max_queue_size)
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("[SUCCESS] Training completed successfully!")
        return history
        
    except Exception as e:
        logger.error(f"[ERROR] Training error: {e}")
        raise

def save_model_artifacts(model, train_gen, history=None):
    """Save all model artifacts required for deployment"""
    
    try:
        logger.info("[SAVE] Saving model artifacts...")
        
        # Load best model if it exists
        if os.path.exists('model/best_model.h5'):
            model = tf.keras.models.load_model('model/best_model.h5')
            logger.info("[SUCCESS] Loaded best model from checkpoint")
        
        # Save main model (CRITICAL: Same format as app.py expects)
        model.save('model/model.h5', save_format='h5')
        logger.info("[SUCCESS] Model saved to model/model.h5")
        
        # CRITICAL: Save labels in EXACT format app.py expects
        labels_dict = train_gen.class_indices
        with open('model/labels.json', 'w') as f:
            json.dump(labels_dict, f, indent=2, sort_keys=True)
        logger.info("[SUCCESS] Labels saved to model/labels.json")
        
        # Save reverse mapping
        reverse_labels = {v: k for k, v in labels_dict.items()}
        with open('model/labels_reverse.json', 'w') as f:
            json.dump(reverse_labels, f, indent=2, sort_keys=True)
        
        # Save training history if available
        if history:
            history_dict = {
                'loss': [float(x) for x in history.history['loss']],
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']]
            }
            
            with open('model/training_history.json', 'w') as f:
                json.dump(history_dict, f, indent=2)
        
        # Save deployment info
        deployment_info = {
            'num_classes': len(labels_dict),
            'input_shape': INPUT_SHAPE,
            'tensorflow_version': tf.__version__,
            'deployment_platform': 'render',
            'optimization_level': 'high',
            'class_names': list(labels_dict.keys())
        }
        
        with open('model/deployment_info.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        logger.info("[SUCCESS] All model artifacts saved successfully")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Error saving model artifacts: {e}")
        return False

def verify_model_compatibility():
    """Verify model is compatible with app.py"""
    
    try:
        logger.info("[VERIFY] Verifying model compatibility...")
        
        # Load and test saved model exactly like app.py does
        test_model = tf.keras.models.load_model('model/model.h5')
        
        # Test with dummy input
        dummy_input = np.random.random((1, 128, 128, 3)).astype(np.float32)
        test_output = test_model.predict(dummy_input, verbose=0)
        
        # Verify output shape
        with open('model/labels.json', 'r') as f:
            labels = json.load(f)
        
        if test_output.shape[1] != len(labels):
            raise ValueError(f"Output shape mismatch: {test_output.shape[1]} vs {len(labels)}")
        
        logger.info("[SUCCESS] Model compatibility verification successful!")
        logger.info(f"[INFO] Model output shape: {test_output.shape}")
        logger.info(f"[INFO] Number of classes: {len(labels)}")
        
        # Cleanup
        del test_model, dummy_input, test_output
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Model compatibility verification failed: {e}")
        return False

def main():
    """Main training function for Render deployment"""
    
    try:
        # Check environment and adjust settings
        batch_size = check_environment()
        
        # Verify data directory
        data_dir = verify_data_directory()
        
        # Create data generators
        train_gen, val_gen = create_optimized_generators(data_dir, batch_size)
        
        # Get number of classes
        num_classes = len(train_gen.class_indices)
        logger.info(f"[SUCCESS] Found {num_classes} classes")
        logger.info(f"[CLASSES] Classes: {list(train_gen.class_indices.keys())}")
        logger.info(f"[SAMPLES] Training samples: {train_gen.samples}")
        logger.info(f"[SAMPLES] Validation samples: {val_gen.samples}")
        
        # Build model
        model = build_lightweight_model(num_classes)
        
        # Display model info
        total_params = model.count_params()
        logger.info(f"[PARAMS] Total parameters: {total_params:,}")
        
        # Train model
        history = train_model_with_checkpoints(model, train_gen, val_gen)
        
        # Evaluate model
        val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
        logger.info(f"[RESULT] Final validation accuracy: {val_accuracy:.4f}")
        logger.info(f"[RESULT] Final validation loss: {val_loss:.4f}")
        
        # Save model artifacts
        if not save_model_artifacts(model, train_gen, history):
            raise RuntimeError("Failed to save model artifacts")
        
        # Verify compatibility
        if not verify_model_compatibility():
            raise RuntimeError("Model compatibility verification failed")
        
        # Success message
        logger.info("\n[SUCCESS] Model training completed successfully!")
        logger.info("[FILES] Files created:")
        logger.info("  - model/model.h5 (main model - READY for app.py)")
        logger.info("  - model/labels.json (class labels)")
        logger.info("  - model/deployment_info.json (deployment info)")
        logger.info(f"\n[READY] Model optimized for Render deployment!")
        logger.info(f"[ACCURACY] Final accuracy: {val_accuracy:.2%}")
        logger.info(f"[PARAMS] Total parameters: {total_params:,}")
        logger.info("\n[DEPLOY] Ready for production deployment!")
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)