import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np

# Configure TensorFlow to use CPU only (same as app.py)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure TensorFlow for CPU-only deployment
tf.config.set_visible_devices([], 'GPU')

print("🚀 Starting model training with CPU-only configuration...")
print("TensorFlow version:", tf.__version__)

# Set data directory
data_dir = 'PlantVillage'
img_size = (128, 128)  # Match app.py image size
batch_size = 32

# Check if data directory exists
if not os.path.exists(data_dir):
    print(f"❌ Error: Data directory '{data_dir}' not found!")
    print("Please ensure your dataset is in the 'PlantVillage' folder")
    exit(1)

print(f"📁 Using data directory: {data_dir}")
print(f"🖼️ Image size: {img_size}")
print(f"📦 Batch size: {batch_size}")

# Enhanced ImageDataGenerator with more augmentation for better generalization
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

# Validation generator without augmentation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

try:
    print("📊 Loading training data...")
    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    print("📊 Loading validation data...")
    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
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

# Build improved CNN model (matching app.py expectations)
print("🏗️ Building CNN model...")

model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    # Second Convolutional Block
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    # Third Convolutional Block
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    # Fully Connected Layers
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compile model with same configuration as app.py
print("⚙️ Compiling model...")
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
print("📋 Model Summary:")
model.summary()

# Enhanced callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.0001,
        verbose=1
    )
]

# Train model
print("🚀 Starting training...")
try:
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )
    
    print("✅ Training completed successfully!")
    
    # Evaluate final model
    print("📊 Final model evaluation:")
    val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

except Exception as e:
    print(f"❌ Training error: {e}")
    exit(1)

# Save model and labels
print("💾 Saving model and labels...")
try:
    os.makedirs('model', exist_ok=True)
    
    # Save model (same format as app.py expects)
    model.save('model/model.h5')
    print("✅ Model saved to model/model.h5")
    
    # Save labels in the exact format app.py expects
    with open('model/labels.json', 'w') as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print("✅ Labels saved to model/labels.json")
    
    # Also create a reverse mapping for verification
    reverse_labels = {v: k for k, v in train_gen.class_indices.items()}
    with open('model/labels_reverse.json', 'w') as f:
        json.dump(reverse_labels, f, indent=2)
    print("✅ Reverse labels saved to model/labels_reverse.json")
    
    # Save training history
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']]
    }
    
    with open('model/training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    print("✅ Training history saved to model/training_history.json")

except Exception as e:
    print(f"❌ Error saving model: {e}")
    exit(1)

print("\n🎉 Model training completed successfully!")
print("📁 Files created:")
print("  - model/model.h5 (main model file)")
print("  - model/labels.json (class labels)")
print("  - model/labels_reverse.json (reverse mapping)")
print("  - model/training_history.json (training metrics)")
print("\n✅ Ready for deployment with app.py!")