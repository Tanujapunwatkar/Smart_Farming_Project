# evaluation/evaluate.py

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

sys.path.append(
    r"C:\Users\tanuj\Downloads\smart_farming_project"
)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

MODEL_PATH = r"C:\Users\tanuj\Downloads\smart_farming_project\model\saved\best_model.keras"
TEST_CSV   = r"C:\Users\tanuj\Downloads\smart_farming_project\data\splits\test.csv"
OUTPUT_DIR = r"C:\Users\tanuj\Downloads\smart_farming_project\evaluation\outputs"
IMG_SIZE   = (64, 64)
BATCH_SIZE = 64
THRESHOLD  = 0.35     # lowered from 0.5 to catch more disease

# ─────────────────────────────────────────────
#  STEP 1 — LOAD MODEL
# ─────────────────────────────────────────────

def load_model(model_path: str) -> tf.keras.Model:
    print("\n📦 STEP 1 — Loading trained model...")
    model = tf.keras.models.load_model(model_path)
    print(f"   Model loaded from {model_path}")
    return model

# ─────────────────────────────────────────────
#  STEP 2 — LOAD TEST DATA
# ─────────────────────────────────────────────

def load_test_dataset(test_csv: str,
                      img_size: tuple,
                      batch_size: int) -> tuple:
    print("\n📂 STEP 2 — Loading test dataset...")
    AUTOTUNE = tf.data.AUTOTUNE

    df     = pd.read_csv(test_csv)
    paths  = df['image_path'].values
    labels = df['label'].values.astype('int32')

    def parse(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3,
                                    expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(parse, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    print(f"   Test images : {len(df)}")
    print(f"   Healthy     : {sum(labels == 0)}")
    print(f"   Diseased    : {sum(labels == 1)}")

    return ds, labels

# ─────────────────────────────────────────────
#  STEP 3 — RUN PREDICTIONS
# ─────────────────────────────────────────────

def predict(model: tf.keras.Model,
            test_ds: tf.data.Dataset,
            threshold: float = THRESHOLD) -> tuple:
    print(f"\n🔍 STEP 3 — Running predictions (threshold={threshold})...")

    probs = model.predict(test_ds, verbose=1)
    probs = probs.flatten()
    preds = (probs >= threshold).astype(int)

    print(f"   Predictions done")
    print(f"   Predicted healthy  : {sum(preds == 0)}")
    print(f"   Predicted diseased : {sum(preds == 1)}")

    return probs, preds

# ─────────────────────────────────────────────
#  STEP 4 — COMPUTE METRICS
# ─────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray) -> dict:
    print("\n📊 STEP 4 — Computing evaluation metrics...")

    metrics = {
        'accuracy' : accuracy_score(y_true, y_pred)  * 100,
        'precision': precision_score(y_true, y_pred) * 100,
        'recall'   : recall_score(y_true, y_pred)    * 100,
        'f1'       : f1_score(y_true, y_pred)        * 100,
        'auc'      : roc_auc_score(y_true, y_prob)
    }

    print("\n" + "=" * 60)
    print("  PHASE 5 — FULL EVALUATION REPORT")
    print("=" * 60)
    print(f"  Threshold  : {THRESHOLD}  (lowered to catch more disease)")
    print(f"  Accuracy   : {metrics['accuracy']:.2f}%")
    print(f"  Precision  : {metrics['precision']:.2f}%")
    print(f"  Recall     : {metrics['recall']:.2f}%")
    print(f"  F1 Score   : {metrics['f1']:.2f}%")
    print(f"  AUC        : {metrics['auc']:.4f}")
    print()
    print("  Classification Report:")
    print("-" * 60)
    print(classification_report(
        y_true, y_pred,
        target_names=['Healthy', 'Diseased'],
        digits=4
    ))
    print("=" * 60)

    # deployment gate
    if metrics['accuracy'] >= 95 and metrics['auc'] >= 0.97:
        print("\n  DEPLOYMENT GATE: PASSED")
        print(f"     Accuracy {metrics['accuracy']:.2f}% >= 95%")
        print(f"     AUC      {metrics['auc']:.4f} >= 0.97")
        print("     Model is ready for deployment!")
        metrics['deploy'] = True
    else:
        print("\n  DEPLOYMENT GATE: NOT YET PASSED")
        print(f"     Accuracy {metrics['accuracy']:.2f}%")
        print(f"     AUC      {metrics['auc']:.4f}")
        print("     Continue tuning before deployment.")
        metrics['deploy'] = False

    return metrics

# ─────────────────────────────────────────────
#  STEP 5 — CONFUSION MATRIX
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          output_dir: str):
    print("\n📊 STEP 5 — Plotting confusion matrix...")
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # raw counts
    sns.heatmap(cm,
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Diseased'],
                yticklabels=['Healthy', 'Diseased'],
                ax=axes[0], linewidths=0.5,
                annot_kws={'size': 16, 'weight': 'bold'})
    axes[0].set_title('Confusion Matrix (counts)',
                      fontweight='bold', fontsize=13)
    axes[0].set_ylabel('True Label', fontsize=11)
    axes[0].set_xlabel('Predicted Label', fontsize=11)

    # percentage
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_pct,
                annot=True, fmt='.1f', cmap='Greens',
                xticklabels=['Healthy', 'Diseased'],
                yticklabels=['Healthy', 'Diseased'],
                ax=axes[1], linewidths=0.5,
                annot_kws={'size': 16, 'weight': 'bold'})
    axes[1].set_title('Confusion Matrix (%)',
                      fontweight='bold', fontsize=13)
    axes[1].set_ylabel('True Label', fontsize=11)
    axes[1].set_xlabel('Predicted Label', fontsize=11)

    plt.suptitle(f'Model Evaluation — Test Set  (threshold={THRESHOLD})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved → {path}")

    total = len(y_true)
    print(f"\n   Confusion Matrix Breakdown:")
    print(f"   TP diseased correctly caught : {tp:>5}  ({tp/total*100:.1f}%)")
    print(f"   TN healthy correctly caught  : {tn:>5}  ({tn/total*100:.1f}%)")
    print(f"   FP healthy wrongly flagged   : {fp:>5}  ({fp/total*100:.1f}%)")
    print(f"   FN diseased missed           : {fn:>5}  ({fn/total*100:.1f}%)")

    if fn > 0:
        fn_pct = fn / total * 100
        print(f"\n   {fn} diseased plants missed ({fn_pct:.1f}%)")
        if fn_pct < 2:
            print("   Excellent — very few missed cases")
        elif fn_pct < 5:
            print("   Acceptable — within safe range for farming")
        else:
            print("   Consider lowering threshold further")

# ─────────────────────────────────────────────
#  STEP 6 — ROC CURVE
# ─────────────────────────────────────────────

def plot_roc_curve(y_true: np.ndarray,
                   y_prob: np.ndarray,
                   auc: float,
                   output_dir: str):
    print("\n📊 STEP 6 — Plotting ROC curve...")

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # find best threshold
    j_scores   = tpr - fpr
    best_idx   = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    best_tpr   = tpr[best_idx]
    best_fpr   = fpr[best_idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#3266ad', linewidth=2.5,
            label=f'ROC Curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linewidth=1,
            linestyle='--', label='Random classifier')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#3266ad')

    # mark best threshold point
    ax.scatter(best_fpr, best_tpr, color='#E24B4A',
               s=120, zorder=5,
               label=f'Best threshold = {best_thresh:.2f}')

    # mark our chosen threshold
    ax.axvline(x=THRESHOLD, color='#3B6D11', linestyle=':',
               linewidth=1.5, label=f'Our threshold = {THRESHOLD}')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve — Healthy vs Diseased',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved → {path}")
    print(f"   Best threshold (Youden J) : {best_thresh:.3f}")

# ─────────────────────────────────────────────
#  STEP 7 — THRESHOLD COMPARISON
# ─────────────────────────────────────────────

def plot_threshold_comparison(y_true: np.ndarray,
                              y_prob: np.ndarray,
                              output_dir: str):
    print("\n📊 STEP 7 — Plotting threshold comparison...")

    thresholds = np.arange(0.1, 0.9, 0.05)
    accuracies, precisions, recalls, f1s, fns = [], [], [], [], []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        accuracies.append(accuracy_score(y_true, preds)   * 100)
        precisions.append(precision_score(y_true, preds,
                          zero_division=0) * 100)
        recalls.append(recall_score(y_true, preds,
                       zero_division=0)    * 100)
        f1s.append(f1_score(y_true, preds,
                   zero_division=0)        * 100)
        cm  = confusion_matrix(y_true, preds)
        fn  = cm[1, 0] if cm.shape == (2, 2) else 0
        fns.append(fn)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # metrics vs threshold
    axes[0].plot(thresholds, accuracies,
                 label='Accuracy',  color='#3266ad', linewidth=2)
    axes[0].plot(thresholds, precisions,
                 label='Precision', color='#3B6D11', linewidth=2)
    axes[0].plot(thresholds, recalls,
                 label='Recall',    color='#D85A30', linewidth=2)
    axes[0].plot(thresholds, f1s,
                 label='F1 Score',  color='#7F77DD', linewidth=2)
    axes[0].axvline(x=THRESHOLD, color='black',
                    linestyle='--', linewidth=1.5,
                    label=f'Our threshold={THRESHOLD}')
    axes[0].set_title('Metrics vs Threshold', fontweight='bold')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Score (%)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # false negatives vs threshold
    axes[1].plot(thresholds, fns,
                 color='#E24B4A', linewidth=2.5)
    axes[1].axvline(x=THRESHOLD, color='black',
                    linestyle='--', linewidth=1.5,
                    label=f'Our threshold={THRESHOLD}')
    axes[1].fill_between(thresholds, fns,
                         alpha=0.15, color='#E24B4A')
    axes[1].set_title('Missed Diseases (FN) vs Threshold',
                      fontweight='bold')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('False Negatives (missed diseases)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Threshold Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(output_dir, 'threshold_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved → {path}")

# ─────────────────────────────────────────────
#  STEP 8 — SAVE REPORT
# ─────────────────────────────────────────────

def save_report(metrics: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'eval_report.txt')

    deploy_status = "READY" if metrics['deploy'] else "NOT READY"

    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  SMART FARMING - FINAL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"  Threshold  : {THRESHOLD}\n")
        f.write(f"  Accuracy   : {metrics['accuracy']:.2f}%\n")
        f.write(f"  Precision  : {metrics['precision']:.2f}%\n")
        f.write(f"  Recall     : {metrics['recall']:.2f}%\n")
        f.write(f"  F1 Score   : {metrics['f1']:.2f}%\n")
        f.write(f"  AUC        : {metrics['auc']:.4f}\n\n")
        f.write(f"  Deployment : {deploy_status}\n")
        f.write("=" * 60 + "\n")

    print(f"\n   Report saved → {path}")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run():
    print("\n" + "=" * 60)
    print("  SMART FARMING - PHASE 5: EVALUATION")
    print("=" * 60)

    model           = load_model(MODEL_PATH)
    test_ds, y_true = load_test_dataset(TEST_CSV, IMG_SIZE, BATCH_SIZE)
    y_prob, y_pred  = predict(model, test_ds, THRESHOLD)
    metrics         = compute_metrics(y_true, y_pred, y_prob)

    plot_confusion_matrix(y_true, y_pred, OUTPUT_DIR)
    plot_roc_curve(y_true, y_prob, metrics['auc'], OUTPUT_DIR)
    plot_threshold_comparison(y_true, y_prob, OUTPUT_DIR)
    save_report(metrics, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  Phase 5 Complete!")
    print("=" * 60)
    print("  Files saved to evaluation/outputs/:")
    print("    - confusion_matrix.png")
    print("    - roc_curve.png")
    print("    - threshold_analysis.png")
    print("    - eval_report.txt")

    if metrics['deploy']:
        print("\n  NEXT STEP: Phase 6 - Deployment!")
    else:
        print("\n  NEXT STEP: Fix model then re-evaluate")

    return metrics


if __name__ == "__main__":
    run()