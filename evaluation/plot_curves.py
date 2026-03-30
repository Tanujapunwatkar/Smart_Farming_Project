# evaluation/plot_curves.py

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

HISTORY_PATH = r"C:\Users\tanuj\Downloads\smart_farming_project\model\saved\training_history.pkl"
OUTPUT_DIR   = r"C:\Users\tanuj\Downloads\smart_farming_project\evaluation\outputs"

# ─────────────────────────────────────────────
#  LOAD HISTORY
# ─────────────────────────────────────────────

def load_history(path: str) -> dict:
    with open(path, 'rb') as f:
        history = pickle.load(f)
    print(f"✅ Loaded training history: {list(history.keys())}")
    return history

# ─────────────────────────────────────────────
#  DETECT OVERFIT / UNDERFIT
# ─────────────────────────────────────────────

def diagnose(history: dict) -> str:
    train_acc = history['accuracy']
    val_acc   = history['val_accuracy']
    train_loss= history['loss']
    val_loss  = history['val_loss']

    final_train_acc = train_acc[-1]  * 100
    final_val_acc   = val_acc[-1]    * 100
    best_val_acc    = max(val_acc)   * 100
    gap             = final_train_acc - final_val_acc

    # check if val loss increased at the end
    val_loss_trend = val_loss[-1] - val_loss[max(0, len(val_loss)-5)]

    print("\n" + "=" * 60)
    print("  OVERFIT / UNDERFIT DIAGNOSIS")
    print("=" * 60)
    print(f"  Final train accuracy  : {final_train_acc:.2f}%")
    print(f"  Final val accuracy    : {final_val_acc:.2f}%")
    print(f"  Best val accuracy     : {best_val_acc:.2f}%")
    print(f"  Train - Val gap       : {gap:.2f}%")
    print(f"  Val loss trend (last5): {val_loss_trend:+.4f}")

    # ── diagnosis logic ──────────────────────────────────────
    if final_train_acc < 80 and final_val_acc < 80:
        diagnosis = "UNDERFIT"
        print("\n  ⚠️  DIAGNOSIS: UNDERFIT")
        print("  Model is not learning well enough.")
        print("  FIXES:")
        print("    → Train for more epochs")
        print("    → Increase model size (more filters)")
        print("    → Reduce Dropout rates")
        print("    → Lower learning rate (try 5e-4)")
        print("    → Increase image size to 128×128")

    elif gap > 8 or (val_loss_trend > 0.05 and final_train_acc > 92):
        diagnosis = "OVERFIT"
        print("\n  ❌ DIAGNOSIS: OVERFIT")
        print(f"  Train-Val gap is {gap:.1f}% — model memorized training data.")
        print("  FIXES:")
        print("    → Increase Dropout (try 0.40–0.50)")
        print("    → Add more augmentation")
        print("    → Increase L2 regularization to 1e-3")
        print("    → Reduce model size")
        print("    → Add more training data")

    elif best_val_acc >= 95:
        diagnosis = "GOOD FIT"
        print("\n  ✅ DIAGNOSIS: GOOD FIT")
        print(f"  Val accuracy {best_val_acc:.2f}% — model is generalizing well!")
        print("  Ready for deployment!")

    elif best_val_acc >= 90:
        diagnosis = "ACCEPTABLE"
        print("\n  🟡 DIAGNOSIS: ACCEPTABLE — slightly below 95%")
        print(f"  Val accuracy {best_val_acc:.2f}% — close but not quite 95%.")
        print("  FIXES:")
        print("    → Train for more epochs")
        print("    → Try image size 128×128")
        print("    → Fine-tune learning rate")

    else:
        diagnosis = "NEEDS WORK"
        print("\n  ⚠️  DIAGNOSIS: NEEDS WORK")
        print(f"  Val accuracy only {best_val_acc:.2f}%")
        print("  FIXES:")
        print("    → Check data quality")
        print("    → Increase model capacity")
        print("    → Try more epochs")

    print("=" * 60)
    return diagnosis

# ─────────────────────────────────────────────
#  PLOT ALL CURVES
# ─────────────────────────────────────────────

def plot_curves(history: dict, diagnosis: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(history['loss']) + 1)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Training Curves  —  Diagnosis: {diagnosis}',
                 fontsize=15, fontweight='bold')

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Plot 1: Accuracy ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.plot(epochs, [a*100 for a in history['accuracy']],
             color='#3266ad', linewidth=2, label='Train accuracy')
    ax1.plot(epochs, [a*100 for a in history['val_accuracy']],
             color='#3B6D11', linewidth=2, linestyle='--',
             label='Val accuracy')
    ax1.axhline(y=95, color='#E24B4A', linestyle=':', linewidth=1.5,
                label='95% target')
    ax1.fill_between(epochs,
                     [a*100 for a in history['accuracy']],
                     [a*100 for a in history['val_accuracy']],
                     alpha=0.1, color='#E24B4A',
                     label=f'Gap (overfit zone)')
    ax1.set_title('Accuracy', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)

    # ── Plot 2: Loss ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0:2])
    ax2.plot(epochs, history['loss'],
             color='#3266ad', linewidth=2, label='Train loss')
    ax2.plot(epochs, history['val_loss'],
             color='#3B6D11', linewidth=2, linestyle='--',
             label='Val loss')
    ax2.set_title('Loss', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: AUC ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, history['auc'],
             color='#7F77DD', linewidth=2, label='Train AUC')
    ax3.plot(epochs, history['val_auc'],
             color='#D85A30', linewidth=2, linestyle='--',
             label='Val AUC')
    ax3.axhline(y=0.97, color='#E24B4A', linestyle=':', linewidth=1.5,
                label='0.97 target')
    ax3.set_title('AUC', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AUC')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)

    # ── Plot 4: Summary card ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')

    best_val_acc  = max(history['val_accuracy']) * 100
    best_val_auc  = max(history['val_auc'])
    final_gap     = (history['accuracy'][-1] -
                     history['val_accuracy'][-1]) * 100
    total_epochs  = len(history['loss'])

    color = '#3B6D11' if best_val_acc >= 95 else \
            '#854F0B' if best_val_acc >= 90 else '#A32D2D'

    summary = (
        f"RESULTS SUMMARY\n\n"
        f"Best val accuracy : {best_val_acc:.2f}%\n"
        f"Best val AUC      : {best_val_auc:.4f}\n"
        f"Train-Val gap     : {final_gap:.2f}%\n"
        f"Epochs run        : {total_epochs}\n\n"
        f"Diagnosis:\n{diagnosis}"
    )

    ax4.text(0.1, 0.9, summary,
             transform=ax4.transAxes,
             fontsize=11, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5',
                       alpha=0.8, edgecolor=color, linewidth=2))

    path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Curves saved → {path}")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run():
    print("\n" + "=" * 60)
    print("  🌱 OVERFIT / UNDERFIT ANALYSIS")
    print("=" * 60)

    history   = load_history(HISTORY_PATH)
    diagnosis = diagnose(history)
    plot_curves(history, diagnosis, OUTPUT_DIR)

    print("\n✅ Analysis complete!")
    print(f"   Chart saved → evaluation/outputs/training_curves.png")
    print("   Open the PNG to visually inspect your curves\n")

    return diagnosis


if __name__ == "__main__":
    run()