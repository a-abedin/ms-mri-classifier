from typing import Tuple, List
import tensorflow as tf
from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras import layers, Model, optimizers


def build_transfer_classifier(
    backbone_name: str = "vgg16",
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    dense_units: int = 64,
    dropout_rate: float = 0.5,
) -> Model:
    """Builds a transfer learning model with frozen backbone and custom classification head."""
    backbone_name_lower = backbone_name.lower()

    if backbone_name_lower == "vgg16":
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    elif backbone_name_lower in ["inception", "inceptionv3"]:
        base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}. Use 'vgg16' or 'inceptionv3'.")

    # Freeze base model weights
    base_model.trainable = False

    x = base_model.output
    x = layers.Dropout(dropout_rate, name="head_dropout_1")(x)
    x = layers.Flatten(name="head_flatten")(x)
    x = layers.Dense(dense_units, activation="relu", name="head_dense")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="prediction")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name=f"MS_{backbone_name_lower}_classifier")
    return model


def get_scheduled_optimizer(
    initial_learning_rate: float = 1e-3,
    decay_steps: int = 1000,
    decay_rate: float = 0.5,
) -> optimizers.Optimizer:
    """Returns an Adam optimizer with ExponentialDecay learning rate schedule."""
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True,
    )
    return optimizers.Adam(learning_rate=lr_schedule)


def get_clinical_metrics() -> List[tf.keras.metrics.Metric]:
    """Returns standard clinical evaluation metrics."""
    return [
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]
