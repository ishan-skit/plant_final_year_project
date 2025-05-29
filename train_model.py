import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np
import gc
from PIL import Image

# CRITICAL: Same CPU configuration as app.py for consistency
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Match app.py
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '2'  # Slightly more for training
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'

# Configure TensorFlow exactly like app.py
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

print("🚀 Starting model training with CPU-only configuration...")
print("TensorFlow version:", tf.__version__)

# CRITICAL: Exact same parameters as app.py expects
data_dir = 'PlantVillage'
img_size = (128, 128)  # MUST match app.py exactly
batch_size = 16  # Reduced for memory efficiency
input_shape = (128, 128, 3)  # Explicit input shape

# Memory optimization
tf.keras.backend.clear_session()
gc.collect()

# Check if data directory exists
if not os.path.exists(data_dir):
    print(f"❌ Error: Data directory '{data_dir}' not found!")
    print("Please ensure your dataset is in the 'PlantVillage' folder")
    exit(1)

print(f"📁 Using data directory: {data_dir}")
print(f"🖼️ Image size: {img_size}")
print(f"📦 Batch size: {batch_size}")

# Enhanced ImageDataGenerator - EXACTLY matching app.py preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,  # CRITICAL: Same normalization as app.py
    validation_split=0.2,
    rotation_range=15,  # Reduced for stability
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode='nearest',
    brightness_range=[0.9, 1.1]  # Minor brightness variation
)

# Validation generator - EXACT same preprocessing as app.py
val_datagen = ImageDataGenerator(
    rescale=1./255,  # CRITICAL: Exact same as app.py
    validation_split=0.2
)

try:
    print("📊 Loading training data...")
    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',  # CRITICAL: Changed to sparse for memory efficiency
        subset='training',
        shuffle=True,
        seed=42,
        interpolation='lanczos'  # Same as PIL.Image.Resampling.LANCZOS in app.py
    )

    print("📊 Loading validation data...")
    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',  # CRITICAL: Changed to sparse
        subset='validation',
        shuffle=False,
        seed=42,
        interpolation='lanczos'
    )

    # Number of output classes
    num_classes = len(train_gen.class_indices)
    print(f"✅ Found {num_classes} classes")
    print("📋 Class labels:", list(train_gen.class_indices.keys()))
    
    # Verify data loading
    print(f"📈 Training samples: {train_gen.samples}")
    print(f"📈 Validation samples: {val_gen.samples}")

except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# Build LIGHTWEIGHT model optimized for CPU deployment
print("🏗️ Building lightweight CNN model...")

# CRITICAL: Smaller model for faster inference on Render
model = Sequential([
    # First Block - Reduced filters
    Conv2D(16, (3,3), activation='relu', input_shape=input_shape, padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    # Second Block
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    # Third Block
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    # Fourth Block - Additional for better feature extraction
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    # Fully Connected Layers - Optimized size
    Flatten(),
    Dense(256, activation='relu'),  # Reduced from 512
    Dropout(0.5),
    Dense(128, activation='relu'),  # Reduced from 256
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# CRITICAL: Exact same compilation as app.py
print("⚙️ Compiling model...")
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',  # Changed for sparse labels
    metrics=['accuracy'],
    run_eagerly=False  # Same as app.py
)

# Display model summary
print("📋 Model Summary:")
model.summary()

# Calculate total parameters
total_params = model.count_params()
print(f"📊 Total parameters: {total_params:,}")

# Enhanced callbacks with model saving
os.makedirs('model', exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',  # Changed to accuracy for better stopping
        patience=7,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=4,
        min_lr=0.00001,
        verbose=1
    ),
    ModelCheckpoint(
        'model/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )
]

# Train model with memory management
print("🚀 Starting training...")
try:
    # Clear any previous session
    tf.keras.backend.clear_session()
    gc.collect()
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=25,  # Increased slightly for better training
        callbacks=callbacks,
        verbose=1,
        workers=1,  # Single worker for stability
        use_multiprocessing=False
    )
    
    print("✅ Training completed successfully!")
    
    # Load best model
    model = tf.keras.models.load_model('model/best_model.h5')
    
    # Evaluate final model
    print("📊 Final model evaluation:")
    val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

except Exception as e:
    print(f"❌ Training error: {e}")
    exit(1)

# CRITICAL: Test model with exact same preprocessing as app.py
print("🧪 Testing model with app.py preprocessing...")
try:
    # Create test image with same preprocessing as app.py
    test_img = np.random.random((128, 128, 3)).astype(np.float32)
    test_img = np.expand_dims(test_img, axis=0)
    
    # Test prediction
    with tf.device('/CPU:0'):
        test_pred = model.predict(test_img, verbose=0)
    
    print(f"✅ Test prediction successful. Output shape: {test_pred.shape}")
    print(f"✅ Prediction confidence: {np.max(test_pred):.4f}")
    
    # Clean up
    del test_img, test_pred
    gc.collect()
    
except Exception as e:
    print(f"❌ Model test failed: {e}")
    exit(1)

# Save model and labels with verification
print("💾 Saving model and labels...")
try:
    # Save main model (CRITICAL: Same format as app.py expects)
    model.save('model/model.h5', save_format='h5')
    print("✅ Model saved to model/model.h5")
    
    # CRITICAL: Save labels in EXACT format app.py expects
    labels_dict = train_gen.class_indices
    with open('model/labels.json', 'w') as f:
        json.dump(labels_dict, f, indent=2, sort_keys=True)
    print("✅ Labels saved to model/labels.json")
    
    # Verify labels format matches app.py expectations
    print("🔍 Verifying labels format...")
    with open('model/labels.json', 'r') as f:
        loaded_labels = json.load(f)
    
    # Create reverse mapping exactly like app.py does
    reverse_labels = {v: k for k, v in loaded_labels.items()}
    print(f"✅ Label verification successful. Classes: {len(reverse_labels)}")
    
    # Save reverse mapping for reference
    with open('model/labels_reverse.json', 'w') as f:
        json.dump(reverse_labels, f, indent=2, sort_keys=True)
    print("✅ Reverse labels saved to model/labels_reverse.json")
    
    # Save training history
    if 'history' in locals():
        history_dict = {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }
        
        with open('model/training_history.json', 'w') as f:
            json.dump(history_dict, f, indent=2)
        print("✅ Training history saved to model/training_history.json")
    
    # Save model info for debugging
    model_info = {
        'num_classes': num_classes,
        'input_shape': input_shape,
        'total_parameters': int(total_params),
        'final_val_accuracy': float(val_accuracy),
        'final_val_loss': float(val_loss),
        'tensorflow_version': tf.__version__,
        'class_names': list(labels_dict.keys())
    }
    
    with open('model/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    print("✅ Model info saved to model/model_info.json")

except Exception as e:
    print(f"❌ Error saving model: {e}")
    exit(1)

# Final verification test
print("🔍 Final verification...")
try:
    # Load and test saved model exactly like app.py does
    test_model = tf.keras.models.load_model('model/model.h5')
    
    # Compile exactly like app.py
    test_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        run_eagerly=False
    )
    
    # Test with dummy input
    dummy_input = np.random.random((1, 128, 128, 3)).astype(np.float32)
    test_output = test_model.predict(dummy_input, verbose=0)
    
    print(f"✅ Final verification successful!")
    print(f"✅ Model output shape: {test_output.shape}")
    print(f"✅ Output matches expected classes: {test_output.shape[1] == num_classes}")
    
    del test_model, dummy_input, test_output
    gc.collect()
    
except Exception as e:
    print(f"❌ Final verification failed: {e}")
    exit(1)

print("\n🎉 Model training completed successfully!")
print("📁 Files created:")
print("  - model/model.h5 (main model file - READY for app.py)")
print("  - model/best_model.h5 (best checkpoint)")
print("  - model/labels.json (class labels - EXACT format for app.py)")
print("  - model/labels_reverse.json (reverse mapping)")
print("  - model/training_history.json (training metrics)")
print("  - model/model_info.json (model information)")
print(f"\n✅ Model optimized for Render deployment!")
print(f"📊 Final accuracy: {val_accuracy:.2%}")
print(f"🔢 Total parameters: {total_params:,}")
print("\n🚀 Ready for deployment with app.py!")