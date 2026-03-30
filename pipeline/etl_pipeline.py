# pipeline/etl_pipeline.py

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")

IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
AUTOTUNE    = tf.data.AUTOTUNE

# ─────────────────────────────────────────────
#  STEP 1 — EXTRACT
#  Read image from disk, validate, decode
# ─────────────────────────────────────────────

def extract(image_path: str) -> np.ndarray:
    """
    Read a single image from disk.
    Raises ValueError if image cannot be read.
    Used for single inference (farmer upload).
    """
    if not os.path.exists(image_path):
        raise ValueError(f"File not found: {image_path}")

    if os.path.getsize(image_path) == 0:
        raise ValueError(f"File is empty: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# ─────────────────────────────────────────────
#  STEP 2 — TRANSFORM
#  Resize + normalize to [0, 1]
# ─────────────────────────────────────────────

def transform(img: np.ndarray,
              img_size: tuple = IMG_SIZE) -> np.ndarray:
    """
    Resize image to target size.
    Normalize pixel values from [0, 255] → [0.0, 1.0].
    """
    img = cv2.resize(img, img_size)
    img = img.astype('float32') / 255.0
    return img

# ─────────────────────────────────────────────
#  STEP 3 — LOAD
#  Add batch dimension for model input
# ─────────────────────────────────────────────

def load_for_inference(img: np.ndarray) -> np.ndarray:
    """
    Add batch dimension.
    (128, 128, 3) → (1, 128, 128, 3)
    Ready to feed directly into model.predict()
    """
    return np.expand_dims(img, axis=0)

# ─────────────────────────────────────────────
#  FULL SINGLE-IMAGE PIPELINE
#  Used when farmer uploads a photo
# ─────────────────────────────────────────────

def etl_single_image(image_path: str) -> np.ndarray:
    """
    Full ETL for one image:
      extract → transform → load
    Returns shape: (1, 128, 128, 3)
    """
    raw   = extract(image_path)
    clean = transform(raw)
    batch = load_for_inference(clean)
    return batch

# ─────────────────────────────────────────────
#  TF.DATA PIPELINE
#  Used for training — fast, memory efficient
# ─────────────────────────────────────────────

def parse_image_label(image_path: tf.Tensor,
                      label: tf.Tensor):
    """
    TensorFlow map function:
      - Read image file
      - Decode JPEG/PNG
      - Resize to 128×128
      - Normalize to [0, 1]
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3,
                                expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augment_train(img: tf.Tensor,
                  label: tf.Tensor):
    """
    On-the-fly augmentation during training only.
    Applied AFTER normalization.
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label

def build_tf_dataset(csv_path: str,
                     batch_size: int = BATCH_SIZE,
                     augment: bool = False,
                     shuffle: bool = False) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from a split CSV file.

    Args:
        csv_path   : path to train.csv / val.csv / test.csv
        batch_size : number of images per batch
        augment    : apply augmentation (train only)
        shuffle    : shuffle order (train only)

    Returns:
        tf.data.Dataset ready for model.fit()
    """
    df      = pd.read_csv(csv_path)
    paths   = df['image_path'].values
    labels  = df['label'].values.astype('int32')

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(paths),
                                  reshuffle_each_iteration=True)

    dataset = dataset.map(parse_image_label,
                          num_parallel_calls=AUTOTUNE)

    if augment:
        dataset = dataset.map(augment_train,
                              num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset

# ─────────────────────────────────────────────
#  BUILD ALL 3 DATASETS
# ─────────────────────────────────────────────

def build_all_datasets(splits_dir: str = SPLITS_DIR,
                       batch_size: int = BATCH_SIZE):
    """
    Build train, val, test tf.data datasets.
    Returns all 3 ready to pass into model.fit()
    """
    train_csv = os.path.join(splits_dir, 'train.csv')
    val_csv   = os.path.join(splits_dir, 'val.csv')
    test_csv  = os.path.join(splits_dir, 'test.csv')

    print("\n🔧 Building tf.data pipelines...")

    train_ds = build_tf_dataset(train_csv,
                                batch_size=batch_size,
                                augment=True,
                                shuffle=True)

    val_ds   = build_tf_dataset(val_csv,
                                batch_size=batch_size,
                                augment=False,
                                shuffle=False)

    test_ds  = build_tf_dataset(test_csv,
                                batch_size=batch_size,
                                augment=False,
                                shuffle=False)

    print(f"   ✅ Train dataset : {len(pd.read_csv(train_csv))} images")
    print(f"   ✅ Val dataset   : {len(pd.read_csv(val_csv))} images")
    print(f"   ✅ Test dataset  : {len(pd.read_csv(test_csv))} images")

    return train_ds, val_ds, test_ds

# ─────────────────────────────────────────────
#  VERIFY PIPELINE
#  Load one batch and check shapes
# ─────────────────────────────────────────────

def verify_pipeline(train_ds: tf.data.Dataset):
    print("\n🔍 Verifying pipeline — loading 1 batch...")

    for images, labels in train_ds.take(1):
        print(f"   Image batch shape : {images.shape}")
        print(f"   Label batch shape : {labels.shape}")
        print(f"   Image dtype       : {images.dtype}")
        print(f"   Label dtype       : {labels.dtype}")
        print(f"   Pixel min / max   : "
              f"{images.numpy().min():.3f} / "
              f"{images.numpy().max():.3f}")
        print(f"   Labels sample     : {labels.numpy()[:8]}")

    print("\n   ✅ Pipeline verified — shapes and values look correct")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run():
    print("\n" + "=" * 60)
    print("  🌱 SMART FARMING — PHASE 3: ETL PIPELINE")
    print("=" * 60)

    train_ds, val_ds, test_ds = build_all_datasets()
    verify_pipeline(train_ds)

    print("\n" + "=" * 60)
    print("  ✅ Phase 3 complete!")
    print("=" * 60)
    print("  tf.data pipeline ready:")
    print("    • train_ds  — shuffled + augmented + prefetched")
    print("    • val_ds    — normalized + batched + prefetched")
    print("    • test_ds   — normalized + batched + prefetched")
    print("\n  Ready for Phase 4 — Model Training\n")

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    run()