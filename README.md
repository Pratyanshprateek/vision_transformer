# Medical Image Classification with ViT

This project contains a Jupyter notebook that trains a pretrained `google/vit-base-patch16-224` Vision Transformer to classify chest X-rays as `NORMAL` or `PNEUMONIA`.

## Files

- `medical_image_classification_vit.ipynb`: end-to-end training, evaluation, visualization, and model saving notebook
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

The notebook is already configured to keep training fast and balanced:

- `TARGET_TOTAL_IMAGES = 3600`
- roughly `1800 NORMAL` + `1800 PNEUMONIA`
- automatic stratified split into `70% train`, `15% validation`, `15% test`

If you want a slightly smaller run, change `TARGET_TOTAL_IMAGES` in the notebook to `3000` or `3200`.

Then open the notebook and run it top to bottom.
