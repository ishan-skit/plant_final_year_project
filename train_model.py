import os
import json
import sys
import logging
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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

# ULTRA AGGRESSIVE CPU-only configuration for Render Free Tier
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'

# Extreme memory optimization
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.experimental.enable_tensor_float_32_execution(False)

# ULTRA MINIMAL configuration - designed for Render Free Tier
DATA_DIR = os.environ.get('DATA_DIR', 'PlantVillage')
IMG_SIZE = (32, 32)  # REDUCED to 32x32 for extreme memory savings
INPUT_SHAPE = (32, 32, 3)  # Must be updated in app.py too
BATCH_SIZE = 4  # Ultra small batch size
EPOCHS = 8  # Minimal epochs
MAX_SAMPLES_PER_CLASS = 100  # Limit samples per class

logger.info(f"[CONFIG] ULTRA MINIMAL - Image: {IMG_SIZE}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}")

def extreme_memory_cleanup():
    """Most aggressive memory cleanup possible"""
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()
    
def check_and_prepare_environment():
    """Check environment with minimal footprint"""
    extreme_memory_cleanup()
    
    if not os.path.exists(DATA_DIR):
        possible_dirs = ['./PlantVillage', '../PlantVillage', '/opt/render/project/src/PlantVillage']
        for alt_dir in possible_dirs:
            if os.path.exists(alt_dir):
                logger.info(f"[FOUND] Using: {alt_dir}")
                return alt_dir
        raise FileNotFoundError(f"Dataset not found")
    
    subdirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if len(subdirs) == 0:
        raise ValueError(f"No classes found in {DATA_DIR}")
    
    logger.info(f"[SUCCESS] Found {len(subdirs)} classes")
    return DATA_DIR

def create_ultra_minimal_generators(data_dir):
    """Create extremely minimal data generators with sample limiting"""
    
    # NO augmentation for memory savings
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    logger.info("[LOADING] Creating minimal generators...")
    
    # Load with class limits to reduce memory
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

def build_micro_model(num_classes):
    """Build the smallest possible working model"""
    
    logger.info("[BUILD] Building MICRO model for 32x32 images...")
    
    # MICRO architecture - absolute minimum
    model = Sequential([
        Input(shape=INPUT_SHAPE),
        
        # Single tiny conv block
        Conv2D(4, (5,5), activation='relu', padding='same'),  # Only 4 filters!
        MaxPooling2D(4,4),  # Aggressive pooling
        Dropout(0.2),
        
        # Second tiny block
        Conv2D(8, (3,3), activation='relu', padding='same'),  # Only 8 filters!
        MaxPooling2D(2,2),
        Dropout(0.3),
        
        # Minimal classifier
        Flatten(),
        Dense(16, activation='relu'),  # Tiny dense layer
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Simple optimizer
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_micro_model(model, train_gen, val_gen):
    """Train with extreme memory conservation"""
    
    os.makedirs('model', exist_ok=True)
    
    # Clean ALL old files
    old_files = ['model/model.h5', 'model/best_model.h5', 'model/model.keras']
    for old_file in old_files:
        if os.path.exists(old_file):
            os.remove(old_file)
            logger.info(f"[CLEANUP] Removed: {old_file}")
    
    # Minimal callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'model/micro_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            save_weights_only=False
        )
    ]

    extreme_memory_cleanup()
    
    logger.info("[TRAINING] Starting micro training...")
    
    # Calculate steps to avoid memory issues
    train_steps = min(train_gen.samples // BATCH_SIZE, 50)  # Limit steps
    val_steps = min(val_gen.samples // BATCH_SIZE, 20)
    
    logger.info(f"[INFO] Training steps: {train_steps}, Validation steps: {val_steps}")
    
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def save_micro_artifacts(train_gen):
    """Save minimal artifacts"""
    
    logger.info("[SAVE] Saving micro artifacts...")
    
    try:
        # Load best model
        if os.path.exists('model/micro_model.h5'):
            model = tf.keras.models.load_model('model/micro_model.h5')
            logger.info("[SUCCESS] Loaded micro model")
        else:
            raise FileNotFoundError("Micro model not found!")
        
        # Save as model.h5 for app.py
        model.save('model/model.h5', save_format='h5')
        logger.info("[SUCCESS] Saved as model.h5")
        
        # Save labels
        labels_dict = train_gen.class_indices
        with open('model/labels.json', 'w') as f:
            json.dump(labels_dict, f, indent=2)
        logger.info("[SUCCESS] Labels saved")
        
        # Update deployment info for 32x32
        deployment_info = {
            'num_classes': len(labels_dict),
            'input_shape': INPUT_SHAPE,
            'image_size': IMG_SIZE,
            'tensorflow_version': tf.__version__,
            'model_type': 'micro_cnn',
            'optimized_for': 'render_free_tier_extreme',
            'class_names': list(labels_dict.keys())
        }
        
        with open('model/deployment_info.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        # Clean temporary files
        if os.path.exists('model/micro_model.h5'):
            os.remove('model/micro_model.h5')
            logger.info("[CLEANUP] Removed temporary file")
        
        del model
        extreme_memory_cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Save failed: {e}")
        return False

def verify_micro_model():
    """Verify micro model works"""
    
    try:
        logger.info("[VERIFY] Testing micro model...")
        
        model = tf.keras.models.load_model('model/model.h5')
        
        # Test with 32x32 input
        dummy_input = np.random.random((1, 32, 32, 3)).astype(np.float32)
        prediction = model.predict(dummy_input, verbose=0)
        
        with open('model/labels.json', 'r') as f:
            labels = json.load(f)
        
        assert prediction.shape[1] == len(labels)
        
        logger.info(f"[SUCCESS] Verification passed!")
        logger.info(f"[INFO] Input: {dummy_input.shape}")
        logger.info(f"[INFO] Output: {prediction.shape}")
        logger.info(f"[INFO] Classes: {len(labels)}")
        
        del model, dummy_input, prediction
        extreme_memory_cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Verification failed: {e}")
        return False

def main():
    """Main function for micro training"""
    
    try:
        logger.info("[START] MICRO MODEL TRAINING - Render Free Tier")
        
        # Prepare
        data_dir = check_and_prepare_environment()
        
        # Create generators
        train_gen, val_gen = create_ultra_minimal_generators(data_dir)
        
        # Model info
        num_classes = len(train_gen.class_indices)
        logger.info(f"[INFO] Classes: {num_classes}")
        logger.info(f"[INFO] Train samples: {train_gen.samples}")
        logger.info(f"[INFO] Val samples: {val_gen.samples}")
        
        # Build micro model
        model = build_micro_model(num_classes)
        
        # Show params
        total_params = model.count_params()
        logger.info(f"[PARAMS] MICRO model: {total_params:,} parameters")
        
        # Train
        history = train_micro_model(model, train_gen, val_gen)
        
        # Evaluate
        val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
        logger.info(f"[RESULT] Final accuracy: {val_accuracy:.4f}")
        
        # Save
        if not save_micro_artifacts(train_gen):
            raise RuntimeError("Save failed")
        
        # Verify
        if not verify_micro_model():
            raise RuntimeError("Verification failed")
        
        # Success
        logger.info("\n" + "="*40)
        logger.info("[SUCCESS] MICRO TRAINING COMPLETED!")
        logger.info("="*40)
        logger.info(f"✅ Model: model/model.h5")
        logger.info(f"✅ Labels: model/labels.json")
        logger.info(f"✅ Accuracy: {val_accuracy:.2%}")
        logger.info(f"✅ Size: {total_params:,} params")
        logger.info(f"✅ Input: 32x32 images")
        logger.info("="*40)
        logger.info("🚀 READY FOR RENDER FREE TIER!")
        
        # IMPORTANT: Update app.py notice
        logger.info("\n⚠️  IMPORTANT: Update app.py:")
        logger.info("   Change IMG_SIZE = (32, 32)")
        logger.info("   Change INPUT_SHAPE = (32, 32, 3)")
        
        return True
        
    except Exception as e:
        logger.error(f"[FATAL] Training failed: {e}")
        return False
    finally:
        extreme_memory_cleanup()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)