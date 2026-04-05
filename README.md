# Medical Image Classification with ViT

This project contains a Jupyter notebook that trains a pretrained `google/vit-base-patch16-224` Vision Transformer to classify chest X-rays as `NORMAL` or `PNEUMONIA`.

## Files

- `medical_image_classification_vit.ipynb`: end-to-end training, evaluation, visualization, and model saving notebook
- `predict_single_xray.py`: run inference on one X-ray image using a saved checkpoint
- `requirements.txt`: Python dependencies

## Dataset

Download the Kaggle **Chest X-Ray Images (Pneumonia)** dataset from:

- [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

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

## Dataset Size

The notebook is currently configured for a stable local run:

- `TARGET_TOTAL_IMAGES = 3000`
- roughly `1500 NORMAL` + `1500 PNEUMONIA`
- automatic stratified split into `70% train`, `15% validation`, `15% test`
- `FORCE_CPU = True` to avoid Apple `mps` kernel crashes seen on some local setups

If you want a larger run later, increase `TARGET_TOTAL_IMAGES` in the notebook and set `FORCE_CPU = False` only if your machine handles `mps` reliably.

Then open the notebook and run it top to bottom.

## Single Image Prediction

After training finishes and the checkpoint exists at `./outputs/vit_chest_xray_classifier.pt`, you can predict one image with:

```bash
source .venv/bin/activate
python predict_single_xray.py chest_xray/test/NORMAL/IM-0001-0001.jpeg --cpu
```

You can also pass your own image path:

```bash
python predict_single_xray.py /path/to/your_xray.jpeg --cpu
```
