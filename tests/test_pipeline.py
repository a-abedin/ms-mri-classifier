import pytest
import numpy as np
import pandas as pd
import tensorflow as tf

from src.dataset_prep import NiftiSliceExtractor
from src.lookup_builder import LookupTableBuilder
from src.models import build_transfer_classifier, get_scheduled_optimizer


def test_safe_normalization_zero_division():
    """Verify that slices with all-zero or flat intensities do not raise zero-division errors."""
    flat_slice = np.ones((100, 100), dtype=np.float32) * 42.0
    norm_img = NiftiSliceExtractor.safe_normalize_to_uint8(flat_slice)
    
    assert norm_img.shape == (100, 100)
    assert norm_img.dtype == np.uint8
    assert np.all(norm_img == 0)


def test_patient_level_splitting_no_leakage():
    """Verify that patient IDs are strictly disjoint across train, val, and test partitions."""
    patient_ids = [f"Patient_{i:02d}" for i in range(1, 31)]
    builder = LookupTableBuilder(raw_base_dir=".", processed_save_dir=".", val_ratio=0.15, test_ratio=0.20)
    
    splits = builder.split_patients(patient_ids)
    train_set = set(splits["train"])
    val_set = set(splits["validation"])
    test_set = set(splits["test"])

    # Verify zero leakage across cohorts
    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0
    assert len(train_set) + len(val_set) + len(test_set) == len(patient_ids)


def test_label_balancing():
    """Verify majority class undersampling results in 1:1 class ratio."""
    records = [
        {"filename": f"img_{i}.png", "label": 0, "partition": "train"} for i in range(70)
    ] + [
        {"filename": f"img_{i}.png", "label": 1, "partition": "train"} for i in range(30)
    ]
    df = pd.DataFrame(records)
    balanced_df = LookupTableBuilder.balance_labels(df)

    counts = balanced_df["label"].value_counts().to_dict()
    assert counts[0] == 30
    assert counts[1] == 30


def test_model_architecture_output_shape():
    """Verify the classification model outputs softmax distributions of shape (batch, 2)."""
    model = build_transfer_classifier(
        backbone_name="vgg16",
        input_shape=(224, 224, 3),
        num_classes=2,
        dense_units=32,
        dropout_rate=0.3,
    )
    
    dummy_input = tf.random.uniform((2, 224, 224, 3), dtype=tf.float32)
    output = model(dummy_input)

    assert output.shape == (2, 2)
    # Check that predictions sum to 1.0 (valid probability distribution)
    np.testing.assert_allclose(tf.reduce_sum(output, axis=-1).numpy(), [1.0, 1.0], atol=1e-5)
