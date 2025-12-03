# How to Access Kaggle SEN12FLOOD Dataset Without Downloading (34 GB)

## Overview
The SEN12FLOOD dataset is 34 GB and too large for most local machines. Here's how to use it efficiently without storing it locally.

---

## Quick Start Guide

### Step 1: Install Kaggle API
```python
pip install kaggle
```

### Step 2: Configure Kaggle Authentication
1. Go to https://www.kaggle.com/settings/account
2. Scroll down to **API** section
3. Click **"Create New API Token"**
4. This downloads `kaggle.json`
5. Place it in your user's `.kaggle` directory:
   - **Windows**: `C:\Users\YourUsername\.kaggle\kaggle.json`
   - **Mac/Linux**: `~/.kaggle/kaggle.json`

### Step 3: Set Permissions (Mac/Linux only)
```bash
chmod 600 ~/.kaggle/kaggle.json
```

---

## Solution Options

### Option 1: Stream Data On-Demand (Recommended)
Download only the files you need, process them, then delete them.

```python
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
import tempfile

api = KaggleApi()
api.authenticate()

# Download specific file to temporary location
api.dataset_download_file(
    'rhythmroy/sen12flood-flood-detection-dataset',
    'file_path_in_dataset',
    path='./temp_data'
)
```

**Pros:**
- ✓ No large disk space needed
- ✓ Access only what you need
- ✓ Works on any machine
- ✓ Cost-effective

**Cons:**
- ✗ Slower (network dependent)
- ✗ Multiple downloads = network usage

---

### Option 2: Use Efficient Data Loaders
Use PyTorch's `DataLoader` with generators to load data in batches without keeping everything in memory.

```python
from torch.utils.data import DataLoader, Dataset

class KaggleFloodDataset(Dataset):
    def __getitem__(self, idx):
        # Load individual image on-demand
        image = load_image(idx)
        mask = load_mask(idx)
        return image, mask

# Load only batch_size images at a time
loader = DataLoader(dataset, batch_size=32)
for batch_images, batch_masks in loader:
    # Train on this batch
    pass
```

**Pros:**
- ✓ Memory efficient
- ✓ Fast local access (if data is downloaded)
- ✓ Standard ML workflow
- ✓ Supports distributed training

**Cons:**
- ✗ Requires downloading full dataset first (34 GB)

---

### Option 3: Use Google Colab (Free GPU)
Google Colab provides built-in Kaggle dataset access.

```python
# Colab automatically mounts Kaggle datasets at /kaggle/input/
import os
os.listdir('/kaggle/input/sen12flood-flood-detection-dataset/')
```

**Pros:**
- ✓ Free GPU access
- ✓ No download needed
- ✓ Fast for training
- ✓ 100 GB temporary storage

**Cons:**
- ✗ Limited training time per session
- ✗ Internet dependent

---

### Option 4: Cloud Storage (AWS S3, GCS)
Upload dataset to cloud and stream from there.

```python
import boto3

s3 = boto3.client('s3')
# Download specific files on-demand
obj = s3.get_object(Bucket='my-bucket', Key='image.tif')
image_data = obj['Body'].read()
```

**Pros:**
- ✓ Scalable to any size
- ✓ High speed transfers
- ✓ Suitable for production

**Cons:**
- ✗ Costs money
- ✗ Setup complexity

---

## Implementation in Your Notebook

The notebook has been updated with:

1. **Cell 1**: Kaggle API installation and configuration
2. **Cell 2**: Download metadata files (CSV only, ~few MB)
3. **Cell 3**: Load and inspect metadata
4. **Cell 4**: Custom data streaming loader class
5. **Cell 5**: Efficient PyTorch DataLoader example

---

## Best Practice Workflow

```
1. Configure Kaggle API ← DO ONCE
   ↓
2. Download only metadata (CSVs) ← Few MB
   ↓
3. Inspect data structure and statistics
   ↓
4. Use DataLoader with generators
   OR
   Download batches on-demand as needed
   ↓
5. Train model
   ↓
6. Save model weights (not dataset)
```

---

## Troubleshooting

### Error: "Kaggle API not authenticated"
**Solution:** Ensure `kaggle.json` is in the correct location and has proper permissions.

```bash
# Check file exists
ls ~/.kaggle/kaggle.json

# Set correct permissions
chmod 600 ~/.kaggle/kaggle.json
```

### Error: "Dataset not found"
**Solution:** Verify dataset name is correct:
```python
api = KaggleApi()
api.authenticate()
api.dataset_list_files('rhythmroy/sen12flood-flood-detection-dataset')
```

### Slow Downloads
**Solution:** 
- Download during off-peak hours
- Use multiple connections (not supported by default Kaggle API)
- Consider Google Colab for faster speeds

---

## File Structure of SEN12FLOOD Dataset

```
SEN12FLOOD/
├── SEN12FLOOD/              (Main data directory)
│   ├── Region_1/
│   │   ├── VV/             (Sentinel-1 VV polarization)
│   │   ├── VH/             (Sentinel-1 VH polarization)
│   │   └── label.tif       (Flood mask)
│   ├── Region_2/
│   └── ...
├── metadata.csv            (Image metadata)
└── README.md
```

---

## Resource Links

- **Kaggle API Docs**: https://github.com/Kaggle/kaggle-api
- **SEN12FLOOD Dataset**: https://www.kaggle.com/datasets/rhythmroy/sen12flood-flood-detection-dataset
- **PyTorch DataLoader**: https://pytorch.org/docs/stable/data.html
- **Google Colab**: https://colab.research.google.com/

---

## Next Steps

1. Run the notebook cells in order
2. Configure your Kaggle API credentials
3. Choose which option works best for your use case
4. Start training your flood detection model!

---

**Happy Deep Learning! 🌊🤖**
