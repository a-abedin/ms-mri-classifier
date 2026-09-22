
import tensorflow as tf
import pandas as pd
from typing import Tuple, Union


class MriDataPipeline:
    """Builds scalable, high-throughput tf.data pipelines for MRI slice datasets."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        num_channels: int = 3,
    ) -> None:
        self.image_size = image_size
        self.num_channels = num_channels

    def parse_png_record(self, filename: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Reads PNG file, decodes, scales to [0, 1] float32, and replicates to 3 channels."""
        file_bytes = tf.io.read_file(filename)
        img = tf.image.decode_png(file_bytes, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)

        # Replicate 1-channel grayscale to 3 channels for ImageNet backbones
        img_3ch = tf.repeat(img, repeats=self.num_channels, axis=-1)
        img_resized = tf.image.resize(img_3ch, self.image_size)

        return img_resized, label

    @staticmethod
    def augment_slice(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Applies spatial real-time data augmentation on training samples."""
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        # Random rotation by multiples of 90 degrees
        num_rotations = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=num_rotations)
        return image, label

    def create_dataset(
        self,
        df_or_csv_path: Union[pd.DataFrame, str],
        batch_size: int = 16,
        is_training: bool = False,
        shuffle_buffer_size: int = 2048,
    ) -> tf.data.Dataset:
        """Constructs an optimized tf.data pipeline with caching and prefetching."""
        if isinstance(df_or_csv_path, str):
            df = pd.read_csv(df_or_csv_path)
        else:
            df = df_or_csv_path.copy()

        filenames = df["filename"].values
        # One-hot encode labels for 2-class softmax classification
        labels = tf.keras.utils.to_categorical(df["label"].values, num_classes=2)

        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

        if is_training:
            dataset = dataset.shuffle(buffer_size=min(len(df), shuffle_buffer_size), reshuffle_each_iteration=True)

        dataset = dataset.map(self.parse_png_record, num_parallel_calls=tf.data.AUTOTUNE)

        if is_training:
            dataset = dataset.map(self.augment_slice, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(batch_size, drop_remainder=is_training)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset
