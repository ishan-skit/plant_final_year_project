import os
import json
import sys
import logging
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# AGGRESSIVE CPU-only configuration for Render Free Tier
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'

# Memory optimization settings
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.experimental.enable_tensor_float_32_execution(False)

# CRITICAL: Optimized configuration for 64x64 images
DATA_DIR = os.environ.get('DATA_DIR', 'PlantVillage')
IMG_SIZE = (64, 64)  # Fixed at 64x64 as requested
INPUT_SHAPE = (64, 64, 3)  # Must match app.py
BATCH_SIZE = 8  # Very small batch size for memory optimization
EPOCHS = 12  # Reduced epochs for faster training

logger.info(f"[CONFIG] Image size: {IMG_SIZE}, Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}")

def aggressive_memory_cleanup():
    """Aggressive memory cleanup for Render"""
    tf.keras.backend.clear_session()
    gc.collect()
    
def check_and_prepare_environment():
    """Check environment and prepare for training"""
    # Clear any existing sessions
    aggressive_memory_cleanup()
    
    # Verify data directory
    if not os.path.exists(DATA_DIR):
        possible_dirs = ['./PlantVillage', '../PlantVillage', '/opt/render/project/src/PlantVillage']
        for alt_dir in possible_dirs:
            if os.path.exists(alt_dir):
                logger.info(f"[FOUND] Using data directory: {alt_dir}")
                return alt_dir
        raise FileNotFoundError(f"Dataset not found in any expected location")
    
    subdirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if len(subdirs) == 0:
        raise ValueError(f"No class directories found in {DATA_DIR}")
    
    logger.info(f"[SUCCESS] Found {len(subdirs)} classes: {subdirs[:5]}{'...' if len(subdirs) > 5 else ''}")
    return DATA_DIR

def create_memory_optimized_generators(data_dir):
    """Create extremely memory-efficient data generators"""
    
    # Minimal augmentation to save memory
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # CRITICAL: Same as app.py
        validation_split=0.2,
        rotation_range=8,
        width_shift_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,  # CRITICAL: Same as app.py
        validation_split=0.2
    )

    logger.info("[LOADING] Creating data generators...")
    
    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='training',
        shuffle=True,
        seed=42
    )

    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation',
        shuffle=False,
        seed=42
    )

    return train_gen, val_gen

def build_ultra_lightweight_model(num_classes):
    """Build ultra-lightweight model for 64x64 images on Render Free"""
    
    logger.info("[BUILD] Building ultra-lightweight model for 64x64 images...")
    
    # Extremely lightweight architecture optimized for 64x64
    model = Sequential([
        Input(shape=INPUT_SHAPE),
        
        # Block 1 - Minimal filters
        Conv2D(8, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Dropout(0.1),
        
        # Block 2 
        Conv2D(16, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Dropout(0.1),
        
        # Block 3
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Dropout(0.2),
        
        # Classifier - Very compact
        Flatten(),
        Dense(64, activation='relu'),  # Small dense layer
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])

    # Compile with same settings as app.py
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_with_memory_optimization(model, train_gen, val_gen):
    """Train model with aggressive memory optimization"""
    
    # Ensure model directory exists and clean old files
    os.makedirs('model', exist_ok=True)
    
    # CRITICAL: Remove old model files as requested
    old_files = ['model/model.h5', 'model/best_model.h5']
    for old_file in old_files:
        if os.path.exists(old_file):
            os.remove(old_file)
            logger.info(f"[CLEANUP] Removed old file: {old_file}")
    
    # Memory-optimized callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.0001,
            verbose=1
        ),
        ModelCheckpoint(
            'model/best_model.h5',  # Using best_model.h5 as requested
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            save_weights_only=False
        )
    ]

    # Clear memory before training
    aggressive_memory_cleanup()
    
    logger.info("[TRAINING] Starting memory-optimized training...")
    
    # Train with minimal memory footprint
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def save_optimized_model_artifacts(train_gen, history=None):
    """Save model artifacts with memory optimization"""
    
    logger.info("[SAVE] Saving optimized model artifacts...")
    
    try:
        # Load the best model saved during training
        if os.path.exists('model/best_model.h5'):
            model = tf.keras.models.load_model('model/best_model.h5')
            logger.info("[SUCCESS] Loaded best model from checkpoint")
        else:
            raise FileNotFoundError("Best model not found!")
        
        # CRITICAL: Save as model.h5 for app.py compatibility
        model.save('model/model.h5', save_format='h5')
        logger.info("[SUCCESS] Model saved as model/model.h5")
        
        # Save labels in exact format app.py expects
        labels_dict = train_gen.class_indices
        with open('model/labels.json', 'w') as f:
            json.dump(labels_dict, f, indent=2)
        logger.info("[SUCCESS] Labels saved")
        
        # Save deployment info
        deployment_info = {
            'num_classes': len(labels_dict),
            'input_shape': INPUT_SHAPE,
            'image_size': IMG_SIZE,
            'tensorflow_version': tf.__version__,
            'optimized_for': 'render_free_tier',
            'class_names': list(labels_dict.keys())
        }
        
        with open('model/deployment_info.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        # Clean up best_model.h5 to save space
        if os.path.exists('model/best_model.h5'):
            os.remove('model/best_model.h5')
            logger.info("[CLEANUP] Removed temporary best_model.h5")
        
        # Memory cleanup
        del model
        aggressive_memory_cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Error saving artifacts: {e}")
        return False

def verify_saved_model():
    """Verify the saved model works correctly"""
    
    try:
        logger.info("[VERIFY] Testing saved model...")
        
        # Test model loading
        test_model = tf.keras.models.load_model('model/model.h5')
        
        # Test prediction with 64x64 dummy input
        dummy_input = np.random.random((1, 64, 64, 3)).astype(np.float32)
        prediction = test_model.predict(dummy_input, verbose=0)
        
        # Verify output
        with open('model/labels.json', 'r') as f:
            labels = json.load(f)
        
        assert prediction.shape[1] == len(labels), f"Shape mismatch: {prediction.shape[1]} vs {len(labels)}"
        
        logger.info(f"[SUCCESS] Model verification passed!")
        logger.info(f"[INFO] Input shape: {dummy_input.shape}")
        logger.info(f"[INFO] Output shape: {prediction.shape}")
        logger.info(f"[INFO] Classes: {len(labels)}")
        
        # Cleanup
        del test_model, dummy_input, prediction
        aggressive_memory_cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Model verification failed: {e}")
        return False

def main():
    """Main training function optimized for Render Free Tier"""
    
    try:
        logger.info("[START] Starting ultra-optimized training for Render Free Tier...")
        
        # Prepare environment
        data_dir = check_and_prepare_environment()
        
        # Create generators
        train_gen, val_gen = create_memory_optimized_generators(data_dir)
        
        # Get classes info
        num_classes = len(train_gen.class_indices)
        logger.info(f"[INFO] Classes: {num_classes}")
        logger.info(f"[INFO] Training samples: {train_gen.samples}")
        logger.info(f"[INFO] Validation samples: {val_gen.samples}")
        
        # Build ultra-lightweight model
        model = build_ultra_lightweight_model(num_classes)
        
        # Show model info
        total_params = model.count_params()
        logger.info(f"[PARAMS] Ultra-lightweight model: {total_params:,} parameters")
        
        # Train with memory optimization
        history = train_with_memory_optimization(model, train_gen, val_gen)
        
        # Get final metrics
        val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
        logger.info(f"[RESULT] Final validation accuracy: {val_accuracy:.4f}")
        
        # Save artifacts
        if not save_optimized_model_artifacts(train_gen, history):
            raise RuntimeError("Failed to save model artifacts")
        
        # Verify model
        if not verify_saved_model():
            raise RuntimeError("Model verification failed")
        
        # Success summary
        logger.info("\n" + "="*50)
        logger.info("[SUCCESS] TRAINING COMPLETED!")
        logger.info("="*50)
        logger.info(f"✅ Model saved: model/model.h5")
        logger.info(f"✅ Labels saved: model/labels.json")
        logger.info(f"✅ Final accuracy: {val_accuracy:.2%}")
        logger.info(f"✅ Model size: {total_params:,} parameters")
        logger.info(f"✅ Optimized for: 64x64 images on Render Free Tier")
        logger.info("="*50)
        logger.info("🚀 READY FOR DEPLOYMENT!")
        
        return True
        
    except Exception as e:
        logger.error(f"[FATAL ERROR] Training failed: {e}")
        return False
    finally:
        # Final cleanup
        aggressive_memory_cleanup()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)