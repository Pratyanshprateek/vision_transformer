# Medical Image Classification with ViT

This project contains a Jupyter notebook that trains a pretrained `google/vit-base-patch16-224` Vision Transformer to classify chest X-rays as `NORMAL` or `PNEUMONIA`.

## Files

- `medical_image_classification_vit.ipynb`: end-to-end training, evaluation, visualization, and model saving notebook
- `predict_single_xray.py`: run inference on one X-ray image using a saved checkpoint
- `requirements.txt`: Python dependencies

## Dataset

Download the Kaggle **Chest X-Ray Images (Pneumonia)** dataset from:

- [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [Kaggel: Chest X-Ray Dataset](https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset)
- [Kaggel: Pediatric Pneumonia Chest X-ray](https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray)

Place the extracted dataset in:

```text
./chest_xray/
  train/NORMAL
  train/PNEUMONIA
  val/NORMAL
  val/PNEUMONIA
  test/NORMAL
  test/PNEUMONIA
```

If you use the Kaggle CLI, a common flow is:

```powershell
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
Expand-Archive chest-xray-pneumonia.zip -DestinationPath .
```

After extraction, make sure the final folder in this project is exactly `./chest_xray`. If Kaggle extracts into a nested folder, move or rename it so the notebook can find it.

## Current Dataset

The project is currently being run with an expanded dataset in this workspace:

- `train/NORMAL = 10941`
- `train/PNEUMONIA = 8868`
- `val/NORMAL = 1758`
- `val/PNEUMONIA = 1428`
- `test/NORMAL = 1169`
- `test/PNEUMONIA = 970`

Total images currently present: `25134`

## Current Training Setup

The notebook is now configured as a V2 pipeline:

- `USE_FULL_DATASET = True`
- full-data stratified split for training/validation/testing
- class imbalance handled with `CrossEntropyLoss(weight=class_weights)`
- stronger training-only augmentation
- `FORCE_CPU = True` for stable Mac execution
- `NUM_WORKERS = 0` on non-CUDA runs

The notebook currently uses `BATCH_SIZE = 32` on CPU. That is an experiment setting, not a guaranteed optimum, so adjust it if runtime or memory behavior becomes poor.

Then open the notebook and run it top to bottom.

## Single Image Prediction

After training finishes and the checkpoint exists at `./outputs/vit_chest_xray_classifier.pt`, you can predict one image with:

macOS / Linux:

```bash
source .venv/bin/activate
python predict_single_xray.py chest_xray/test/NORMAL/IM-0001-0001.jpeg --cpu
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python predict_single_xray.py "C:\path\to\your_xray.jpeg" --cpu
```

Windows Command Prompt:

```bat
.venv\Scripts\activate
python predict_single_xray.py C:\path\to\your_xray.jpeg --cpu
```

You can also pass your own image path:

```bash
python predict_single_xray.py /path/to/your_xray.jpeg --cpu
```
