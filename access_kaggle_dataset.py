"""
Quick Script: Access Kaggle SEN12FLOOD Dataset Without Downloading Full 34 GB

This script demonstrates three approaches to work with the dataset without 
storing it all locally.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from io import BytesIO
import tempfile

# ============================================================================
# APPROACH 1: Download individual files on-demand
# ============================================================================

def approach_1_stream_data():
    """
    Download only specific files as needed, process, then delete.
    Perfect for: Exploratory analysis, sampling, testing
    """
    print("\n" + "="*70)
    print("APPROACH 1: Stream Individual Files On-Demand")
    print("="*70)
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Installing kaggle API...")
        os.system(f"{sys.executable} -m pip install kaggle -q")
        from kaggle.api.kaggle_api_extended import KaggleApi
    
    api = KaggleApi()
    api.authenticate()
    
    dataset_name = 'rhythmroy/sen12flood-flood-detection-dataset'
    
    # List files in dataset
    files = api.dataset_list_files(dataset_name)
    print(f"\nTotal files in dataset: {len(files.files)}")
    
    # Show first few files
    print("\nFirst 10 files:")
    for i, file in enumerate(files.files[:10]):
        print(f"  {i+1}. {file.name} ({file.size / 1024**3:.2f} GB)")
    
    # Download specific metadata file temporarily
    print("\n✓ To download a specific file:")
    print("  api.dataset_download_file(")
    print("      'rhythmroy/sen12flood-flood-detection-dataset',")
    print("      'path/to/file.csv',")
    print("      path='./temp'")
    print("  )")


# ============================================================================
# APPROACH 2: Efficient PyTorch DataLoader
# ============================================================================

def approach_2_dataloader():
    """
    Use PyTorch DataLoader with generators for memory-efficient batch loading.
    Perfect for: Training, large dataset processing, distributed training
    """
    print("\n" + "="*70)
    print("APPROACH 2: Efficient PyTorch DataLoader")
    print("="*70)
    
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        print("Installing torch...")
        os.system(f"{sys.executable} -m pip install torch -q")
        import torch
        from torch.utils.data import Dataset, DataLoader
    
    from PIL import Image
    
    class FloodDataset(Dataset):
        def __init__(self, image_dir, mask_dir):
            self.images = sorted(Path(image_dir).glob('*.png'))
            self.mask_dir = Path(mask_dir)
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            # Load on-demand (not all at once!)
            img = Image.open(self.images[idx])
            mask = Image.open(self.mask_dir / self.images[idx].name)
            return torch.FloatTensor(np.array(img)), torch.LongTensor(np.array(mask))
    
    print("""
    # Create dataset that loads images on-demand
    dataset = FloodDataset('images_dir', 'masks_dir')
    
    # Create loader for batch processing (e.g., 32 images at a time)
    loader = DataLoader(dataset, batch_size=32, num_workers=4)
    
    # Use in training loop
    for batch_images, batch_masks in loader:
        # Process only current batch (memory efficient!)
        train_step(batch_images, batch_masks)
    
    Benefits:
    - Only loads batch_size images in memory at a time
    - num_workers=4 loads next batch while GPU processes current batch
    - Perfect for 34 GB dataset on small machine
    """)


# ============================================================================
# APPROACH 3: Google Colab (Free GPU, Automatic Dataset Access)
# ============================================================================

def approach_3_colab():
    """
    Use Google Colab for free GPU and automatic Kaggle access.
    Perfect for: Training, GPU access, temporary storage
    """
    print("\n" + "="*70)
    print("APPROACH 3: Google Colab (Free GPU)")
    print("="*70)
    
    print("""
    # Run this in Google Colab (colab.research.google.com)
    
    # Step 1: Mount Kaggle
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    # Step 2: Dataset is automatically mounted at /kaggle/input/
    import os
    files = os.listdir('/kaggle/input/sen12flood-flood-detection-dataset/')
    
    # Step 3: Use normally (100 GB free storage per session!)
    import pandas as pd
    df = pd.read_csv('/kaggle/input/...../metadata.csv')
    
    Benefits:
    - Free GPU (Tesla K80 or better)
    - 100 GB temporary storage
    - Kaggle datasets pre-mounted
    - No setup needed
    
    Limitations:
    - 12 hour max session length
    - Internet dependent
    - Resets after session ends
    """)


# ============================================================================
# APPROACH 4: Download Only Metadata
# ============================================================================

def approach_4_metadata_only():
    """
    Download only CSV metadata files, not images.
    Perfect for: EDA, planning, data analysis without full download
    """
    print("\n" + "="*70)
    print("APPROACH 4: Download Metadata Only")
    print("="*70)
    
    print("""
    # Metadata files are typically small (< 100 MB)
    
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    # Download specific CSV files only
    api.dataset_download_file(
        'rhythmroy/sen12flood-flood-detection-dataset',
        'metadata.csv',  # or whatever CSV files exist
        path='./metadata'
    )
    
    # Analyze metadata to decide what to download
    import pandas as pd
    df = pd.read_csv('./metadata/metadata.csv')
    
    print(f"Dataset has {len(df)} samples")
    print(f"Regions: {df['region'].unique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Now you can selectively download only needed regions/dates
    
    Benefits:
    - Metadata is tiny (< 100 MB vs 34 GB)
    - Understand dataset before downloading
    - Plan your training strategy
    - Filter by region, date, etc.
    """)


# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================

def print_recommendations():
    """Print recommendations based on use case"""
    print("\n" + "="*70)
    print("RECOMMENDATIONS BY USE CASE")
    print("="*70)
    
    recommendations = {
        "Exploratory Data Analysis": {
            "Best": "Approach 4 (Metadata Only)",
            "Why": "Understand dataset without downloading 34 GB"
        },
        "Model Development": {
            "Best": "Approach 2 (PyTorch DataLoader)",
            "Why": "Train efficiently without large disk space"
        },
        "Production Training": {
            "Best": "Approach 3 (Colab) or Cloud Storage",
            "Why": "Unlimited resources, high speed"
        },
        "Quick Testing": {
            "Best": "Approach 1 (Stream On-Demand)",
            "Why": "Download only what you test"
        },
        "Offline Work": {
            "Best": "Approach 2 (Download once, use DataLoader)",
            "Why": "Balance: download once, memory efficient"
        }
    }
    
    for use_case, rec in recommendations.items():
        print(f"\n{use_case}:")
        print(f"  → {rec['Best']}")
        print(f"  → {rec['Why']}")


# ============================================================================
# SETUP INSTRUCTIONS
# ============================================================================

def print_setup_instructions():
    """Print setup instructions for Kaggle API"""
    print("\n" + "="*70)
    print("SETUP: Configure Kaggle API (One-time only)")
    print("="*70)
    
    print("""
    1. Go to https://www.kaggle.com/settings/account
    
    2. Scroll to "API" section at bottom
    
    3. Click "Create New API Token"
       → This downloads kaggle.json
    
    4. Place kaggle.json in correct location:
       Windows: C:\\Users\\YourUsername\\.kaggle\\kaggle.json
       Mac/Linux: ~/.kaggle/kaggle.json
    
    5. Set permissions (Mac/Linux only):
       chmod 600 ~/.kaggle/kaggle.json
    
    6. Verify:
       python -c "from kaggle.api.kaggle_api_extended import KaggleApi; 
                   api = KaggleApi(); api.authenticate(); print('✓ Success')"
    """)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║     SEN12FLOOD Dataset Access Without Downloading Full 34 GB           ║
    ║                                                                        ║
    ║  Dataset: https://www.kaggle.com/datasets/rhythmroy/sen12flood-      ║
    ║           flood-detection-dataset                                     ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Setup
    print_setup_instructions()
    
    # Show all approaches
    try:
        approach_1_stream_data()
    except Exception as e:
        print(f"Note: Approach 1 requires Kaggle API configured: {e}")
    
    approach_2_dataloader()
    approach_3_colab()
    approach_4_metadata_only()
    
    # Recommendations
    print_recommendations()
    
    print("\n" + "="*70)
    print("GET STARTED:")
    print("="*70)
    print("""
    1. Configure Kaggle API (see instructions above)
    2. Choose an approach that fits your use case
    3. Check the Jupyter notebook for detailed examples
    4. Start with metadata analysis before downloading full dataset
    """)
