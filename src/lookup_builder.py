import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.dataset_prep import NiftiSliceExtractor


class LookupTableBuilder:
    """Constructs patient-separated, balanced lookup CSV tables for MRI datasets."""

    def __init__(
        self,
        raw_base_dir: str,
        processed_save_dir: str,
        planes: Optional[List[str]] = None,
        val_ratio: float = 0.15,
        test_ratio: float = 0.20,
        random_seed: int = 42,
    ) -> None:
        self.raw_base_dir = raw_base_dir
        self.processed_save_dir = processed_save_dir
        self.planes = planes or ["sagittal", "coronal", "axial"]
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.extractor = NiftiSliceExtractor()

    def split_patients(self, patient_ids: List[str]) -> Dict[str, List[str]]:
        """Splits patient identifiers strictly across train, validation, and test subsets."""
        train_val, test = train_test_split(
            patient_ids, test_size=self.test_ratio, random_state=self.random_seed
        )
        adjusted_val_ratio = self.val_ratio / (1.0 - self.test_ratio)
        train, val = train_test_split(
            train_val, test_size=adjusted_val_ratio, random_state=self.random_seed
        )
        return {"train": train, "validation": val, "test": test}

    @staticmethod
    def balance_labels(df: pd.DataFrame) -> pd.DataFrame:
        """Balances positive and negative slice samples by undersampling the majority class."""
        pos_df = df[df["label"] == 1]
        neg_df = df[df["label"] == 0]
        min_len = min(len(pos_df), len(neg_df))

        if min_len == 0:
            return df

        balanced = pd.concat([pos_df.iloc[:min_len], neg_df.iloc[:min_len]])
        return balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

    def process_patient_volume(
        self,
        patient_id: str,
        dataset_name: str,
        image_path: str,
        mask_path: str,
        partition: str,
    ) -> List[Dict]:
        """Loads patient NIfTI pair, exports valid slices, and returns metadata records."""
        records = []
        image_vol = self.extractor.load_nifti_data(image_path)
        mask_vol = self.extractor.load_nifti_data(mask_path)

        for plane in self.planes:
            slice_gen = self.extractor.extract_slices_from_volume(
                image_vol, mask_vol, plane=plane
            )
            for idx, img_arr, label in slice_gen:
                filename = f"{dataset_name}_{patient_id}_{plane}_{idx}.png"
                out_path = os.path.join(
                    self.processed_save_dir, partition, plane, filename
                )
                self.extractor.save_slice_as_png(img_arr, out_path)

                records.append({
                    "filename": out_path,
                    "subject": patient_id,
                    "dataset_name": dataset_name,
                    "partition": partition,
                    "label": label,
                    "view": plane,
                })
        return records

    def export_csv(self, records: List[Dict], output_csv_path: str, balance: bool = True) -> pd.DataFrame:
        """Exports extracted slice records to a standardized CSV file."""
        df = pd.DataFrame(records)
        if balance:
            df = df.groupby("partition", group_keys=False).apply(self.balance_labels)
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        return df
