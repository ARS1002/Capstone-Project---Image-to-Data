# Image-to-Data

Extract product attributes – weight, dimensions, voltage, wattage, volume – and their units straight from product photos, for e-commerce listings that lack text descriptions. Built as a team of 4 (team lead) for the Amazon ML Challenge, then developed into our capstone project.

**Result:** F1-score of **0.73** on the test data with a hybrid OCR + CNN + BERT model. The full write-up is in [REPORT.md](REPORT.md).

## What's in this repo

| Step | File | What it does |
|---|---|---|
| 1 | `src/download_images.py` | Downloads product images listed in `dataset/train.csv` (parallel, with retries) |
| 2 | `src/process_images.py` | Converts images to RGB and resizes them to 128×128 |
| 3 | `src/train.py` | Trains a CNN (JAX/Flax) that predicts the value (regression) and its unit (classification) together |
| – | `src/sanity.py` | Checks that a predictions file uses only allowed units and the right format |
| – | `src/constants.py` | Allowed units for each attribute (e.g. `item_weight` → gram, kilogram, pound…) |
| – | `src/test.ipynb` | Loads the data, runs the format check and a sample image download |

`src/main.py` runs steps 1–3 in order.

## How to run

```bash
pip install -r requirements.txt
# put the challenge CSVs in dataset/ (see dataset/README.md)
python src/main.py
```

Check a predictions file:

```bash
python src/sanity.py --test_filename dataset/sample_test.csv --output_filename dataset/sample_test_out.csv
```

## Approach (summary of REPORT.md)

- OCR pulls raw text from each image
- A CNN extracts visual features; BERT embeds the OCR text for context
- Multi-task learning predicts the numeric value and its unit at the same time

The scripts here cover the data pipeline and the CNN baseline. The OCR and BERT parts are described in the report.

## Tech

Python · JAX / Flax · Optax · TensorFlow / Keras · Pandas · Pillow
