# Setup & Quickstart — FusionSegNet v5

## Prerequisites

- A Google account with Google Colab access
- ~3 GB of free space in Google Drive (for dataset + model weights)
- nuScenes mini dataset downloaded from [nuscenes.org](https://www.nuscenes.org/nuscenes)

---
 
## Step 0 — Download the nuScenes Dataset (Required for Both Options)
 
1. Go to [https://www.nuscenes.org/nuscenes](https://www.nuscenes.org/nuscenes)
2. Click **"Download"** and create a free account if you don't have one
3. Under **"nuScenes"**, download **`v1.0-mini`** — the file is called `v1.0-mini.tgz` (~3.9 GB)
 
> The mini split has 10 scenes (~400 samples). It is enough to train and evaluate the full pipeline.
 
---
 
---
 
 
### A1. Upload Dataset to Google Drive
 
1. Open [https://drive.google.com](https://drive.google.com)
2. Upload `v1.0-mini.tgz` directly to **My Drive** (the root, not inside any subfolder)
 
Your Drive should look like this:
 
```
My Drive/
└── v1.0-mini.tgz          <-- place it here, not inside a folder
```
 
The notebook expects it at exactly this location. If you place it elsewhere, update this line in **Cell 0**:
 
```python
DATA_TGZ = '/content/drive/MyDrive/v1.0-mini.tgz'
#                                   ^^^^^^^^^^^
#                     Change this if your file is in a subfolder, e.g.:
#                     '/content/drive/MyDrive/datasets/v1.0-mini.tgz'
```
 
---
 
### A2. Upload the Notebook to Google Drive
 
1. Download `FusionSegNet_v5.ipynb` from this repository
2. Upload it anywhere in your Google Drive (e.g. `My Drive/FusionSegNet/`)
3. Right-click the file → **Open with → Google Colaboratory**
 
---
 
### A3. Enable GPU Runtime
 
In Colab, go to:
 
```
Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save
```
 
> Without a GPU, training will take 10-15x longer. Always use GPU for this notebook.
 
---
 
### A4. Run the Notebook — Step by Step
 
**Important:** After Cell 1 (Install Dependencies), you must restart the Colab runtime before continuing.
 
#### Cell 0 — Environment Detection
No action needed. Automatically detects Colab and sets all paths.
 
---
 
#### Cell 1 — Install Dependencies
Click **Run**. Wait for it to complete (~2 minutes).
 
When it finishes, you will see:
```
All dependencies installed.
```
 
**Now restart the runtime:**
```
Runtime → Restart session
```
or use the dropdown next to "Run all":
```
Run all → Restart session and run all
```
 
> If you use "Restart session and run all", Colab will automatically re-run everything from the top including re-installing dependencies. This is fine and is the recommended approach.
 
---
 
#### Cell 2 — Mount Google Drive
A popup will ask for Drive permissions. Click **Connect to Google Drive** and follow the prompts. Your Drive will be mounted at `/content/drive/MyDrive/`.
 
---
 
#### Cell 3 — Setup Paths
This cell:
- Creates output directories at `/content/outputs/`
- Extracts `v1.0-mini.tgz` from Drive to `/content/data/` (only if not already extracted)
 
> Extraction takes ~3-5 minutes. The extracted dataset stays in Colab's temporary storage for the duration of your session. If your session disconnects, it will need to be re-extracted next time.
 
**If extraction fails**, check that `DATA_TGZ` points to the correct Drive path:
```python
DATA_TGZ = '/content/drive/MyDrive/v1.0-mini.tgz'
```
 
---
 
#### Cell 4 — Import FusionSegNet Package
Imports all model classes, dataset utilities, and training functions.
 
---
 
#### Cell 5 — Sanity Check
Loads one sample and generates one mask. You should see a class distribution table and a visualisation with 3 panels (Camera / Label Map / Overlay).
 
Expected output:
```
Class distribution:
  Background  : 180432 px  (35.2%)
  Road        :  92048 px  (18.0%)
  Sidewalk    :  15820 px   (3.1%)
  ...
Confidence mean : 0.621
```
 
If this cell errors, the most common causes are:
- Wrong `DATAROOT` path (the extracted dataset is not where the notebook expects it)
- Missing `LIDAR_TOP` data (make sure you downloaded the full mini split, not just images)
 
---
 
#### Cell 6 — Generate Masks + Confidence Maps
Runs the multi-sensor fusion pipeline over all 6 cameras × all samples.
 
**Expected time: 10–20 minutes.** A progress bar shows the status. Files are saved to:
```
/content/outputs/masks/         mask_<token>_<cam>.png
/content/outputs/confidence/    conf_<token>_<cam>.npy
```
 
Already-generated files are skipped on re-run.
 
---
 
#### Cell 7 — Extract Copy-Paste Crops
Extracts vehicle and pedestrian patches for augmentation. Saves ~120 crops per class to:
```
/content/outputs/paste_crops/
```
 
---
 
#### Cell 8 — Train
**Expected time: 2–3 hours on T4 GPU for 45 epochs.**
 
The model is saved to Google Drive after every epoch that improves validation mIoU:
```
/content/drive/MyDrive/best_fusionsegnet_v5.pth
```
 
Training is saved to Drive so it survives session disconnections. If your session dies, you can resume — the next run will reload the best checkpoint automatically.
 
Training output per epoch:
```
Ep  1/45 | Loss:1.2341 | mIoU:0.3241 | BinIoU:0.4821 | Bac:0.512  Roa:0.421  ...
  Checkpoint saved (mIoU=0.3241) -> /content/drive/MyDrive/best_fusionsegnet_v5.pth
```
 
---
 
#### Cells 9–14 — Evaluate, TTA, FPS, Export, Visualise
Run these after training completes. Each is independent and can be re-run without re-training.
 
---

| Cells | What happens |
|---|---|
| 1 (Install deps) | Installs all Python packages then RESTART THE RUNTIME(from run all dropdown menue) IN GOOGLE COLAB |
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
- ~2–3 hours on a T4 GPU depending on Colab availability

---

 
### A5. Output Files (Colab)
 
| File | Location | Description |
|---|---|---|
| `best_fusionsegnet_v5.pth` | Google Drive root | Best model weights by mIoU |
| `best_fusionsegnet_v5_scripted.pt` | Google Drive root | TorchScript export |
| `masks/` | `/content/outputs/masks/` | PNG pseudo-label masks |
| `confidence/` | `/content/outputs/confidence/` | Float16 confidence maps (.npy) |
| `visualizations/` | `/content/outputs/visualizations/` | Overlay images |
 
> Weights are saved to Drive (persistent). Everything in `/content/` is lost when the session ends.
 
---

