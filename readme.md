# MCPP — Myocardia Prediction Project

Place Python code in `src/mcpp/`, including an empty `__init__.py`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 -m ipykernel install --user --name mcpp --display-name "Python (MCPP)"
```

In VS Code, install the Python and Jupyter extensions, open a notebook,
and select **Python (MCPP)** from the kernel picker.

Import the package with `import mcpp`.

Store datasets in `data/`; this directory is ignored by Git.

include data files in /data/ as
labeled_test_data
test_samples
training_data