# MCPP — Myocardia Prediction Project

Place Python code in the `src/mcpp/` package.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 -m ipykernel install --user --name mcpp --display-name "Python (MCPP)"
```

In VS Code, install the Python and Jupyter extensions, open a notebook,
and select **Python (MCPP)** from the kernel picker. This kernel uses the
repository's `.venv` environment. Restart the notebook kernel after installing
the package so it picks up the editable installation.

Import the package with `import mcpp`.

Store datasets in `data/`; this directory is ignored by Git.

include data files in /data/ as
labeled_test_data
test_samples
training_data
