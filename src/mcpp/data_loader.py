import numpy as np
from pathlib import Path
import pandas as pd
print("hello!")

class DataSet():
    def __init__(self, name, labelled=False):
        current_dir = Path(__file__).resolve()
        root_dir = current_dir.parent.parent.parent
        dataset_dir = root_dir / "data" / name
        self.name = name
        self.directory = dataset_dir
        self.data = self.load_csv()
        self.patients = self.load_patients()

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

        
class Patient():
    def __init__(self, patient_num, data_row, sample_dir):
        self.data_row = data_row
        print(data_row)
        self.patient_num = patient_num
        label = data_row['MI'] if 'MI' in data_row.index else ''
        self.sample_file = sample_dir / f'{patient_num}{label}.npy'
        print(f"Loading {self.sample_file}")
        self.data = np.load(self.sample_file)

    

    
