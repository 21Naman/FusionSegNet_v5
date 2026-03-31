# Setup & Quickstart

## Prerequisites

- A Google account with Google Colab access
- ~3 GB of free space in Google Drive (for dataset + model weights)
- nuScenes mini dataset downloaded from [nuscenes.org](https://www.nuscenes.org/nuscenes)

---

## Step 1 — Download the Dataset

1. Register at [nuscenes.org](https://www.nuscenes.org/nuscenes)
2. Download `v1.0-mini.tgz` (approximately 4.2 GB)
3. Upload it to the root of your Google Drive (not inside any folder)

---

## Step 2 — Open the Notebook in Colab

1. Upload `FusionSegNet_v5.ipynb` to your Google Drive
2. Right-click the file and select **Open with > Google Colaboratory**
3. In Colab: go to **Runtime > Change runtime type** and select **T4 GPU**

---

## Step 3 — Run the Notebook

Run cells in order from top to bottom:

| Cells | What happens |
|---|---|
| 1 (Install deps) | Installs all Python packages |
| 2 (Mount Drive) | Mounts your Google Drive |
| 3 (Setup Paths) | Extracts the nuScenes dataset, creates output directories |
| 4 (FusionSegmenter) | Defines the multi-sensor pseudo-label generator |
| 5 (Sanity Check) | Verifies dataset loads and one mask generates correctly |
| 6 (Generate Masks) | Runs the full pseudo-label generation (~10–20 min) |
| 7 (Copy-Paste Crops) | Extracts vehicle and pedestrian patches for augmentation |
| 8 (Architecture) | Defines the FusionSegNet model |
| 9 (Loss Functions) | Defines Confidence-Weighted CE, Dice, Focal losses |
| 10 (Dataset) | Defines the dataset class with confidence map loading |
| 11 (Temporal Loss) | Defines the temporal consistency loss |
| 12 (Training Loop) | **Trains the model** — this is the main training cell (~45 epochs) |
| 13 (Evaluation) | Evaluates on the full dataset, prints mIoU |
| 14 (TTA Evaluation) | 5-view TTA evaluation for best reported mIoU |
| 15 (FPS Benchmark) | Measures inference FPS at multiple resolutions and batch sizes |
| 16 (Export) | Exports model to TorchScript for deployment |
| 17 (Visualisation) | Generates qualitative prediction overlays |
| 18 (Summary Card) | Prints final results summary |
| 19 (Ablation) | Prints the ablation study table (fill in your numbers) |
| 20 (Confidence Maps) | Visualises the sensor-agreement confidence maps |
| 21 (Writeup Points) | Ready-to-use hackathon submission text |

---

## Expected Training Time

- nuScenes mini, all 6 cameras, 45 epochs
- ~2–4 hours on a T4 GPU depending on Colab availability

---

## Output Files

After training, the following are saved to Google Drive or `/content/outputs/`:

| File | Description |
|---|---|
| `best_fusionsegnet_v5.pth` | Best checkpoint by validation mIoU |
| `best_fusionsegnet_v5_scripted.pt` | TorchScript export for deployment |
| `outputs/masks/` | PNG pseudo-label masks for all samples |
| `outputs/confidence/` | Float16 NumPy confidence maps |
| `outputs/visualizations/` | Qualitative overlay images |

---

## Local Setup (Optional)

If you prefer to run locally instead of Colab:

```bash
git clone https://github.com/YOUR_USERNAME/FusionSegNet.git
cd FusionSegNet
pip install -r requirements.txt
jupyter notebook FusionSegNet_v5.ipynb
```

Note: you will need to adjust the file paths at the top of the path setup cell (remove the Google Drive paths and point to your local dataset location).
