import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
# import mcpp.visualisation as vis
print("hello!")

class DataSet():
    def __init__(self, name, train_prop=0.6, val_prop=0.2, test_prop=0.2, labelled=False):
        current_dir = Path(__file__).resolve()
        root_dir = current_dir.parent.parent.parent
        dataset_dir = root_dir / "data" / name
        self.name = name
        self.directory = dataset_dir
        self.data = self.load_csv()
        self.patients = self.load_patients()
        self.create_subsets(train_prop, val_prop, test_prop)

    def load_csv(self, n=None):
        csv_files = list(self.directory.glob('*.csv'))
        csv_file_name = csv_files[0].name if csv_files else None
        data = pd.read_csv(self.directory / csv_file_name)
        return data

    def load_patients(self, n=None):
        print("Loading patients")
        print(self.directory)

        subdirs = [item.name for item in self.directory.iterdir() if item.is_dir()]
        print(subdirs)
        only_subdir = subdirs if len(subdirs) == 1 else None
        if len(subdirs) == 1:
            only_subdir = subdirs[0]
            # Count only files (excluding folders) inside the subdirectory
            file_count = sum(1 for item in (self.directory / only_subdir).iterdir() if item.is_file())
        else:
            raise Exception("Expected only one sub-directory in dataset directory")

        self.num_patients = file_count

        patients = []
        for i in range(self.num_patients):
            patients.append(Patient(i, self.data.iloc[i], self.directory / only_subdir))

        return patients

        
    def create_subsets(self, train_prop=0.6, val_prop=0.2, test_prop=0.2):
        proportions = np.array([train_prop, val_prop, test_prop])
        if not np.isfinite(proportions).all() or (proportions <= 0).any() or not np.isclose(proportions.sum(), 1):
            raise ValueError("Split proportions must be positive and sum to 1.")
        indices = np.arange(self.num_patients)
        labels = self.data["MI"] if "MI" in self.data else None
        train_val, test = train_test_split(
            indices, test_size=test_prop, random_state=42, stratify=labels)
        train, val = train_test_split(
            train_val, test_size=val_prop / (train_prop + val_prop),
            random_state=42, stratify=labels.iloc[train_val] if labels is not None else None)
        self.train = DataSubSet(self, train)
        self.val = DataSubSet(self, val)
        self.test = DataSubSet(self, test)

    def cross_validation(self, n_splits=5):
        # Only train + val enter the folds; test stays held out.
        indices = np.concatenate([self.train.indices, self.val.indices])
        labels = self.data.iloc[indices]["MI"] if "MI" in self.data else None
        if labels is not None and labels.value_counts().min() < n_splits:
            raise ValueError("Each class needs at least n_splits non-test patients.")
        splitter = StratifiedKFold if labels is not None else KFold
        folds = splitter(n_splits=n_splits, shuffle=True, random_state=42)
        return [(DataSubSet(self, indices[train], fold), DataSubSet(self, indices[val], fold))
                for fold, (train, val) in enumerate(folds.split(indices, labels))]


class DataSubSet():
    def __init__(self, dataset, indices, cross_val_num=None):
        self.cross_val_num = cross_val_num
        self.indices = np.asarray(indices)
        self.data = dataset.data.iloc[self.indices]
        self.patients = [dataset.patients[i] for i in self.indices]


class Patient():
    def __init__(self, patient_num, data_row, sample_dir):
        self.data_row = data_row
        print(data_row)
        self.patient_num = patient_num
        label = data_row['MI'] if 'MI' in data_row.index else ''
        self.sample_file = sample_dir / f'{patient_num}{label}.npy'
        print(f"Loading {self.sample_file}")
        self.data = np.load(self.sample_file)


    
