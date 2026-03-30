# model/cnn_model.py

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers # type: ignore

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

IMG_SIZE = (128, 128, 3)

# ─────────────────────────────────────────────
#  CNN ARCHITECTURE
# ─────────────────────────────────────────────

def build_cnn(input_shape: tuple = IMG_SIZE) -> tf.keras.Model:
    """
    Custom CNN for binary plant disease classification.

    Architecture:
      3 × Conv blocks  (Conv → BN → ReLU → MaxPool → Dropout)
      1 × Conv block   (deeper features)
      GlobalAveragePooling
      Dense head → sigmoid output

    Design decisions:
      - BatchNormalization  : stabilizes training, faster convergence
      - Dropout             : prevents overfitting
      - GlobalAveragePooling: fewer params than Flatten, less overfit
      - L2 regularization  : extra penalty on large weights
      - sigmoid output     : binary probability [0=healthy, 1=diseased]
    """

    model = models.Sequential([

        # ── Input ──────────────────────────────────────────
        layers.Input(shape=input_shape),

        # ── Block 1 : 32 filters ───────────────────────────
        layers.Conv2D(32, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # ── Block 2 : 64 filters ───────────────────────────
        layers.Conv2D(64, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # ── Block 3 : 128 filters ──────────────────────────
        layers.Conv2D(128, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.30),

        # ── Block 4 : 256 filters ──────────────────────────
        layers.Conv2D(256, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.30),

        # ── Classifier head ────────────────────────────────
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.50),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.30),

        # ── Output ─────────────────────────────────────────
        # sigmoid → value between 0 and 1
        # < 0.5  → healthy
        # >= 0.5 → diseased
        layers.Dense(1, activation='sigmoid')
    ])

    return model

def compile_model(model: tf.keras.Model,
                  learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    Compile model with:
      - Adam optimizer
      - Binary crossentropy loss (binary classification)
      - Metrics: accuracy, AUC, precision, recall
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model

def get_model_summary(model: tf.keras.Model):
    """Print full model summary with parameter counts."""
    model.summary()
    total   = model.count_params()
    trainable = sum([tf.size(w).numpy()
                     for w in model.trainable_weights])
    print(f"\n  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🌱 SMART FARMING — CNN MODEL ARCHITECTURE")
    print("=" * 60)

    model = build_cnn()
    model = compile_model(model)
    get_model_summary(model)

    print("\n  ✅ Model built and compiled successfully")
    print("  Input shape  : (128, 128, 3)")
    print("  Output shape : (1,) — sigmoid probability")
    print("  0.0 → healthy   |   1.0 → diseased\n")