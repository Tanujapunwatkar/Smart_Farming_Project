# model/train.py  — CPU OPTIMIZED

import os
import sys
import pickle
import tensorflow as tf

sys.path.append(
    r"C:\Users\tanuj\Downloads\smart_farming_project"
)

from model.cnn_model       import compile_model
from pipeline.etl_pipeline import build_all_datasets

# ─────────────────────────────────────────────
#  CPU OPTIMIZED CONFIG
# ─────────────────────────────────────────────

SPLITS_DIR    = r"C:\Users\tanuj\Downloads\smart_farming_project\data\splits"
MODEL_DIR     = r"C:\Users\tanuj\Downloads\smart_farming_project\model\saved"

BATCH_SIZE    = 64       # larger batch = fewer steps per epoch = faster
EPOCHS        = 30       # reduced from 50
LEARNING_RATE = 1e-3
IMG_SIZE      = (64, 64) # smaller image = much faster on CPU

CLASS_WEIGHT  = {
    0: 25409 / (2 * 8000),
    1: 25409 / (2 * 17409)
}

# ─────────────────────────────────────────────
#  CPU OPTIMIZATIONS
# ─────────────────────────────────────────────

def set_cpu_optimizations():
    # use all available CPU cores
    num_cores = os.cpu_count()
    tf.config.threading.set_intra_op_parallelism_threads(num_cores)
    tf.config.threading.set_inter_op_parallelism_threads(num_cores)
    print(f"   Using {num_cores} CPU cores")

# ─────────────────────────────────────────────
#  SMALLER CNN FOR CPU
# ─────────────────────────────────────────────

def build_cpu_cnn(input_shape=(64, 64, 3)):
    """
    Lightweight CNN optimized for CPU training.
    Smaller image (64x64) + fewer filters = 10x faster.
    Still deep enough to learn good features.
    """
    from tensorflow.keras import layers, models, regularizers # type: ignore

    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Block 1 — 32 filters
        layers.Conv2D(32, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Block 2 — 64 filters
        layers.Conv2D(64, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Block 3 — 128 filters
        layers.Conv2D(128, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.30),

        # Block 4 — 256 filters
        layers.Conv2D(256, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.30),

        # Classifier head
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.40),

        # Output
        layers.Dense(1, activation='sigmoid')
    ])

    return model

# ─────────────────────────────────────────────
#  CPU-FRIENDLY tf.data PIPELINE
# ─────────────────────────────────────────────

def build_cpu_datasets(splits_dir, batch_size, img_size):
    """
    Rebuild tf.data pipeline with smaller image size (64x64).
    Cache dataset in memory for faster repeated reads.
    """
    import pandas as pd
    AUTOTUNE = tf.data.AUTOTUNE

    def parse(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3,
                                    expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.15)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label

    def make_ds(csv_path, shuffle=False, augment_flag=False):
        df      = pd.read_csv(csv_path)
        paths   = df['image_path'].values
        labels  = df['label'].values.astype('int32')
        total   = len(df)

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=total,
                            reshuffle_each_iteration=True)
        ds = ds.map(parse, num_parallel_calls=AUTOTUNE)
        if augment_flag:
            ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.cache()        # cache in RAM — much faster after epoch 1
        ds = ds.prefetch(AUTOTUNE)
        return ds, total

    print("\n📦 Building CPU-optimized tf.data pipelines...")
    print(f"   Image size : {img_size[0]}×{img_size[1]} (CPU friendly)")
    print(f"   Batch size : {batch_size}")

    train_ds, n_train = make_ds(
        os.path.join(splits_dir, 'train.csv'),
        shuffle=True, augment_flag=True)

    val_ds, n_val = make_ds(
        os.path.join(splits_dir, 'val.csv'),
        shuffle=False, augment_flag=False)

    test_ds, n_test = make_ds(
        os.path.join(splits_dir, 'test.csv'),
        shuffle=False, augment_flag=False)

    print(f"   ✅ Train : {n_train} images")
    print(f"   ✅ Val   : {n_val} images")
    print(f"   ✅ Test  : {n_test} images")

    return train_ds, val_ds, test_ds

# ─────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────

def get_callbacks(model_dir):
    os.makedirs(model_dir, exist_ok=True)

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor              = 'val_auc',
            patience             = 8,
            restore_best_weights = True,
            mode                 = 'max',
            verbose              = 1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = 0.5,
            patience = 3,
            min_lr   = 1e-7,
            verbose  = 1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath       = os.path.join(model_dir, 'best_model.keras'),
            monitor        = 'val_auc',
            save_best_only = True,
            mode           = 'max',
            verbose        = 1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(model_dir, 'training_log.csv')
        )
    ]

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run():
    print("\n" + "=" * 60)
    print("  🌱 SMART FARMING — PHASE 4: MODEL TRAINING (CPU)")
    print("=" * 60)

    # CPU optimizations
    print("\n⚙️  Setting CPU optimizations...")
    set_cpu_optimizations()

    # datasets
    train_ds, val_ds, test_ds = build_cpu_datasets(
        SPLITS_DIR, BATCH_SIZE, IMG_SIZE
    )

    # model
    print("\n🔧 Building lightweight CNN for CPU...")
    model = build_cpu_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model = compile_model(model, learning_rate=LEARNING_RATE)
    model.summary()

    print(f"\n⚖️  Class weights:")
    print(f"   Healthy  (0) : {CLASS_WEIGHT[0]:.3f}")
    print(f"   Diseased (1) : {CLASS_WEIGHT[1]:.3f}")

    print(f"\n🚀 Starting training...")
    print(f"   Image size    : {IMG_SIZE[0]}×{IMG_SIZE[1]}")
    print(f"   Batch size    : {BATCH_SIZE}")
    print(f"   Max epochs    : {EPOCHS}")
    print(f"   Early stop    : patience=8 on val_auc")
    print("-" * 60)

    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = EPOCHS,
        class_weight    = CLASS_WEIGHT,
        callbacks       = get_callbacks(MODEL_DIR),
        verbose         = 1
    )

    # save history
    os.makedirs(MODEL_DIR, exist_ok=True)
    history_path = os.path.join(MODEL_DIR, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)

    # results
    best_val_acc = max(history.history['val_accuracy']) * 100
    best_val_auc = max(history.history['val_auc'])
    epochs_run   = len(history.history['loss'])

    print("\n" + "=" * 60)
    print("  ✅ Phase 4 Training Complete!")
    print("=" * 60)
    print(f"  Epochs run        : {epochs_run}")
    print(f"  Best val accuracy : {best_val_acc:.2f}%")
    print(f"  Best val AUC      : {best_val_auc:.4f}")
    print(f"  Model saved       → model/saved/best_model.keras")
    print(f"  Training log      → model/saved/training_log.csv")

    if best_val_acc >= 95.0:
        print("\n  🎉 Accuracy ≥ 95% — Ready for Phase 5 Evaluation!")
    else:
        print(f"\n  ⚠️  Accuracy = {best_val_acc:.2f}% — Phase 5 will diagnose")

    print("\n  Ready for Phase 5 — Evaluation\n")

    return model, history


if __name__ == "__main__":
    run()