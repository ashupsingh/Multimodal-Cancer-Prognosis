"""
Download reference dataset, splits, and pretrained models from Mahmood Lab Google Drive.
"""

import os
import zipfile
import gdown

DATA_FILES = {
    'gbmlgg_splits': {
        'id': '1AjB8oqkjgWeXj-9nWvec7Ce8zAOtkOCs',
        'output': './data/TCGA_GBMLGG/splits.zip',
        'extract_to': './data/TCGA_GBMLGG/splits'
    },
    'gbmlgg_data': {
        'id': '1McOK93l31ALaoAJaE8vJGRB3S3NK99MU',
        'output': './data/TCGA_GBMLGG/TCGA_GBMLGG.zip',
        'extract_to': './data/TCGA_GBMLGG'
    },
    'gbmlgg_checkpoints_surv': {
        'id': '1kjSsEDGYR5AaeBQ7K8LybWuAva3gl_oW',
        'output': './checkpoints/TCGA_GBMLGG/surv_15_rnaseq.zip',
        'extract_to': './checkpoints/TCGA_GBMLGG'
    },
    'gbmlgg_checkpoints_grad': {
        'id': '1aV7RaSMAoAWzl1BJceY2HvGLy3AItXBB',
        'output': './checkpoints/TCGA_GBMLGG/grad_15.zip',
        'extract_to': './checkpoints/TCGA_GBMLGG'
    }
}


def download_and_extract(item_key, force=False):
    item = DATA_FILES[item_key]
    out_dir = os.path.dirname(item['output'])
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(item['output']) or force:
        print(f"Downloading {item_key} (ID: {item['id']}) to {item['output']}...")
        gdown.download(id=item['id'], output=item['output'], quiet=False)
    else:
        print(f"File {item['output']} already exists. Skipping download.")

    if os.path.exists(item['output']):
        print(f"Extracting {item['output']} to {item['extract_to']}...")
        os.makedirs(item['extract_to'], exist_ok=True)
        with zipfile.ZipFile(item['output'], 'r') as zip_ref:
            zip_ref.extractall(item['extract_to'])
        print(f"Extracted {item_key} successfully.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='gbmlgg_splits',
                        choices=list(DATA_FILES.keys()) + ['all'])
    args = parser.parse_args()

    if args.target == 'all':
        for k in DATA_FILES.keys():
            download_and_extract(k)
    else:
        download_and_extract(args.target)
