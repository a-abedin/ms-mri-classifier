import os
import argparse
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

from src.tf_pipeline import MriDataPipeline
from src.models import build_transfer_classifier, get_scheduled_optimizer, get_clinical_metrics
from src.evaluate import ClinicalEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate MS MRI Classification Model.")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to YAML config.")
    parser.add_argument("--train-csv", type=str, required=True, help="Path to training lookup CSV.")
    parser.add_argument("--val-csv", type=str, required=True, help="Path to validation lookup CSV.")
    parser.add_argument("--test-csv", type=str, required=True, help="Path to test lookup CSV.")
    parser.add_argument("--backbone", type=str, default=None, choices=["vgg16", "inceptionv3"])
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    backbone = args.backbone or cfg["training"].get("backbone", "inceptionv3")
    batch_size = cfg["training"].get("batch_size", 16)
    epochs = cfg["training"].get("epochs", 60)
    image_size = tuple(cfg["data"].get("image_size", [224, 224]))

    print(f"[*] Configuration Loaded. Selected Backbone: {backbone.upper()}")

    # 1. Build TensorFlow Data Pipelines
    pipeline = MriDataPipeline(image_size=image_size, num_channels=3)
    train_ds = pipeline.create_dataset(args.train_csv, batch_size=batch_size, is_training=True)
    val_ds = pipeline.create_dataset(args.val_csv, batch_size=batch_size, is_training=False)
    test_ds = pipeline.create_dataset(args.test_csv, batch_size=batch_size, is_training=False)

    # 2. Build Model
    model = build_transfer_classifier(
        backbone_name=backbone,
        input_shape=(image_size[0], image_size[1], 3),
        num_classes=cfg["architecture"].get("num_classes", 2),
        dense_units=cfg["architecture"].get("dense_units", 64),
        dropout_rate=cfg["architecture"].get("dropout_rate", 0.5),
    )

    optimizer = get_scheduled_optimizer(
        initial_learning_rate=cfg["training"].get("initial_lr", 1e-3),
        decay_steps=1000,
        decay_rate=cfg["training"].get("decay_rate", 0.5),
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=get_clinical_metrics(),
    )
    model.summary()

    # 3. Callbacks
    os.makedirs("artifacts/checkpoints", exist_ok=True)
    os.makedirs("artifacts/logs", exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg["training"].get("early_stopping_patience", 12),
            restore_best_weights=True,
            verbose=1,
        ),
        CSVLogger(f"artifacts/logs/{backbone}_training_log.csv"),
        ModelCheckpoint(
            filepath=f"artifacts/checkpoints/{backbone}_best_model.h5",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    # 4. Train Model
    print("[*] Starting Model Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # 5. Inference and Evaluation on Test Set
    print("[*] Performing Diagnostic Inference on Independent Test Cohort...")
    y_true_list = []
    for _, labels in test_ds:
        y_true_list.append(labels.numpy())
    y_true = np.concatenate(y_true_list, axis=0)

    y_pred_probs = model.predict(test_ds)

    evaluator = ClinicalEvaluator(output_dir="artifacts/plots")
    metrics = evaluator.compute_metrics(y_true, y_pred_probs)

    print("\n" + "=" * 50)
    print("        INDEPENDENT CLINICAL TEST SET METRICS     ")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<24}: {v:.4f}")
    print("=" * 50 + "\n")

    # 6. Generate and Export Plots
    evaluator.plot_confusion_matrix(y_true, y_pred_probs, save_filename=f"{backbone}_confusion_matrix.png")
    evaluator.plot_roc_curve(y_true, y_pred_probs, save_filename=f"{backbone}_roc_curves.png")
    evaluator.plot_precision_recall_curve(y_true, y_pred_probs, save_filename=f"{backbone}_pr_curves.png")
    evaluator.plot_training_history(history.history, save_filename=f"{backbone}_training_history.png")

    print("[+] All evaluation plots generated successfully in 'artifacts/plots/'")


if __name__ == "__main__":
    main()
