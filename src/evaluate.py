import os
from typing import Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class ClinicalEvaluator:
    """Computes comprehensive diagnostic metrics and generates publication-grade plots."""

    def __init__(
        self,
        class_names: Optional[Tuple[str, str]] = None,
        output_dir: str = "artifacts/plots",
    ) -> None:
        self.class_names = class_names or ("Non-Lesion (Control)", "MS Lesion")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Calculates standard clinical evaluation metrics."""
        y_pred_binary = (y_pred_probs[:, 1] >= threshold).astype(int)
        y_true_binary = (
            np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true.astype(int)
        )

        acc = accuracy_score(y_true_binary, y_pred_binary)
        prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, 1])
        roc_auc = auc(fpr, tpr)

        metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall_sensitivity": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
        }
        return metrics

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
        threshold: float = 0.5,
        save_filename: str = "confusion_matrix.png",
    ) -> None:
        """Plots and saves an annotated confusion matrix heatmap."""
        y_pred = (y_pred_probs[:, 1] >= threshold).astype(int)
        y_true_idx = (
            np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true.astype(int)
        )

        cm = confusion_matrix(y_true_idx, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)

        tick_marks = np.arange(len(self.class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(self.class_names, rotation=25, ha="right")
        ax.set_yticklabels(self.class_names)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12,
                    fontweight="bold",
                )

        ax.set_title("Clinical Diagnostic Confusion Matrix", fontsize=13, pad=12)
        ax.set_ylabel("True Ground Truth Label", fontsize=11)
        ax.set_xlabel("Predicted Classifier Label", fontsize=11)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(out_path)
        plt.close(fig)

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
        save_filename: str = "roc_curves.png",
    ) -> None:
        """Computes and renders multi-class and micro/macro averaged ROC curves."""
        n_classes = 2
        y_true_ohe = (
            y_true
            if y_true.ndim > 1
            else tf.keras.utils.to_categorical(y_true, num_classes=n_classes)
        )

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_ohe[:, i], y_pred_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_true_ohe.ravel(), y_pred_probs.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        ax.plot(
            fpr[0],
            tpr[0],
            color="black",
            lw=1.8,
            label=f"ROC curve of class 0 (AUC = {roc_auc[0]:.2f})",
        )
        ax.plot(
            fpr[1],
            tpr[1],
            color="#2ca02c",
            lw=1.8,
            label=f"ROC curve of class 1 (AUC = {roc_auc[1]:.2f})",
        )
        ax.plot(
            fpr["micro"],
            tpr["micro"],
            color="#e377c2",
            linestyle=":",
            lw=2.5,
            label=f"Micro-average ROC (AUC = {roc_auc['micro']:.2f})",
        )
        ax.plot([0, 1], [0, 1], "k--", lw=1.2)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
        ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, pad=12)
        ax.legend(loc="lower right", frameon=True)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(out_path)
        plt.close(fig)

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
        save_filename: str = "precision_recall_curves.png",
    ) -> None:
        """Plots Precision-Recall curves indicating balance under class skew."""
        n_classes = 2
        y_true_ohe = (
            y_true
            if y_true.ndim > 1
            else tf.keras.utils.to_categorical(y_true, num_classes=n_classes)
        )

        precision = dict()
        recall = dict()
        avg_precision = dict()

        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(
                y_true_ohe[:, i], y_pred_probs[:, i]
            )
            avg_precision[i] = average_precision_score(
                y_true_ohe[:, i], y_pred_probs[:, i]
            )

        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_true_ohe.ravel(), y_pred_probs.ravel()
        )
        avg_precision["micro"] = average_precision_score(
            y_true_ohe, y_pred_probs, average="micro"
        )

        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        ax.plot(
            recall[0],
            precision[0],
            color="black",
            lw=1.8,
            label=f"PR curve class 0 (AP = {avg_precision[0]:.3f})",
        )
        ax.plot(
            recall[1],
            precision[1],
            color="#2ca02c",
            lw=1.8,
            label=f"PR curve class 1 (AP = {avg_precision[1]:.3f})",
        )
        ax.plot(
            recall["micro"],
            precision["micro"],
            color="#1f77b4",
            linestyle=":",
            lw=2.5,
            label=f"Micro-average PR (AP = {avg_precision['micro']:.3f})",
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
        ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
        ax.set_title("Precision-Recall Curve", fontsize=13, pad=12)
        ax.legend(loc="lower left", frameon=True)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(out_path)
        plt.close(fig)

    def plot_training_history(
        self,
        history_dict: Dict,
        save_filename: str = "training_history.png",
    ) -> None:
        """Plots training vs validation accuracy and loss across epochs."""
        epochs = range(1, len(history_dict["loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

        # Accuracy
        ax1.plot(epochs, history_dict["accuracy"], label="Training Accuracy", color="#1f77b4", lw=2)
        if "val_accuracy" in history_dict:
            ax1.plot(epochs, history_dict["val_accuracy"], label="Validation Accuracy", color="#ff7f0e", lw=2)
        ax1.set_title("Training and Validation Accuracy", fontsize=12)
        ax1.set_xlabel("Epochs", fontsize=10)
        ax1.set_ylabel("Accuracy", fontsize=10)
        ax1.legend(loc="lower right")
        ax1.grid(alpha=0.3)

        # Loss
        ax2.plot(epochs, history_dict["loss"], label="Training Loss", color="#1f77b4", lw=2)
        if "val_loss" in history_dict:
            ax2.plot(epochs, history_dict["val_loss"], label="Validation Loss", color="#ff7f0e", lw=2)
        ax2.set_title("Training and Validation Loss", fontsize=12)
        ax2.set_xlabel("Epochs", fontsize=10)
        ax2.set_ylabel("Cross-Entropy Loss", fontsize=10)
        ax2.legend(loc="upper right")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(out_path)
        plt.close(fig)
