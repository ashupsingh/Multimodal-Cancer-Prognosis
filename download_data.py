"""
Download the reference dataset, splits and checkpoints of Pathomic Fusion
from the Mahmood Lab Google Drive folder.

Targets:
    gbmlgg_splits            15-fold CV split files (136 MB)
    gbmlgg_graphs            cell graphs + split files + clinical tables only
                             (~0.3 GB, fetched out of the 14 GB data archive
                             with HTTP range requests; enough to train every
                             model from the pre-extracted histology embeddings)
    gbmlgg_data              full data archive incl. ROI images (14.4 GB)
    gbmlgg_checkpoints_surv  authors' survival checkpoints
    gbmlgg_checkpoints_grad  authors' grade checkpoints

All downloads resume after an interruption and retry on network errors.
"""

import io
import os
import sys
import time
import zlib
import struct
import zipfile
import argparse
import requests

DRIVE_URL = ('https://drive.usercontent.google.com/download'
             '?id={id}&export=download&confirm=t')

DATA_FILES = {
    'gbmlgg_splits': {
        'id': '1AjB8oqkjgWeXj-9nWvec7Ce8zAOtkOCs',
        'output': './data/TCGA_GBMLGG/splits.zip',
        'extract_to': './data/TCGA_GBMLGG/splits'
    },
    'gbmlgg_data': {
        'id': '1McOK93l31ALaoAJaE8vJGRB3S3NK99MU',
        'output': './data/TCGA_GBMLGG/TCGA_GBMLGG.zip',
        'extract_to': './data'
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


# ── Resumable HTTP range download ─────────────────────────────────────

def remote_size(url):
    """Total size in bytes of a remote file."""
    with requests.get(url, headers={'Range': 'bytes=0-0'}, stream=True,
                      timeout=60) as r:
        r.raise_for_status()
        return int(r.headers['Content-Range'].split('/')[-1])


def download_range(url, dest, start=0, end=None, max_failures=100):
    """
    Download bytes [start, end) of `url` into `dest`.

    An existing partial `dest` is continued, and network errors are retried,
    so the call can simply be repeated after an interruption.
    """
    if end is None:
        end = remote_size(url)
    total = end - start
    failures = 0
    while True:
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if have >= total:
            return dest
        try:
            headers = {'Range': f'bytes={start + have}-{end - 1}'}
            with requests.get(url, headers=headers, stream=True,
                              timeout=60) as r:
                if r.status_code != 206:
                    raise RuntimeError(f'unexpected HTTP status {r.status_code}')
                with open(dest, 'ab') as f:
                    last = time.time()
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        failures = 0
                        if time.time() - last > 60:
                            last = time.time()
                            print(f"  {os.path.basename(dest)}: "
                                  f"{f.tell() / 1e6:.0f} / {total / 1e6:.0f} MB",
                                  flush=True)
        except Exception as exc:
            failures += 1
            if failures > max_failures:
                raise
            print(f"  network error ({type(exc).__name__}); retry {failures}",
                  flush=True)
            time.sleep(min(60, 5 * failures))


def download_and_extract(item_key, force=False):
    """Download one archive completely and extract it."""
    item = DATA_FILES[item_key]
    os.makedirs(os.path.dirname(item['output']), exist_ok=True)
    if force and os.path.exists(item['output']):
        os.remove(item['output'])

    if not os.path.exists(item['output']):
        print(f"Downloading {item_key} to {item['output']} ...")
        part = item['output'] + '.part'
        download_range(DRIVE_URL.format(id=item['id']), part)
        os.replace(part, item['output'])
    else:
        print(f"File {item['output']} already exists. Skipping download.")

    print(f"Extracting {item['output']} to {item['extract_to']} ...")
    os.makedirs(item['extract_to'], exist_ok=True)
    with zipfile.ZipFile(item['output'], 'r') as zip_ref:
        zip_ref.extractall(item['extract_to'])
    print(f"Extracted {item_key} successfully.")


# ── Selective download from the data archive ──────────────────────────

class _RemoteTail(io.RawIOBase):
    """Read-only view of a remote file that only holds its last bytes
    (enough for `zipfile` to read the central directory)."""

    def __init__(self, url, size, tail_bytes=32 << 20):
        self.size, self.pos = size, 0
        self.tail_start = max(0, size - tail_bytes)
        tmp = os.path.join('.', f'.zip_tail_{os.getpid()}.tmp')
        download_range(url, tmp, self.tail_start, size)
        with open(tmp, 'rb') as f:
            self.tail = f.read()
        os.remove(tmp)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (offset if whence == 0 else
                    self.pos + offset if whence == 1 else self.size + offset)
        return self.pos

    def read(self, n=-1):
        if self.pos < self.tail_start:
            raise IOError('requested bytes outside the downloaded tail')
        o = self.pos - self.tail_start
        data = self.tail[o:] if n < 0 else self.tail[o:o + n]
        self.pos += len(data)
        return data


def _extract_from_range(blob_path, base_offset, members, dataroot,
                        strip='TCGA_GBMLGG/'):
    """Extract zip members from a downloaded byte range of the archive."""
    with open(blob_path, 'rb') as f:
        for m in members:
            if m.is_dir():
                continue
            f.seek(m.header_offset - base_offset)
            header = struct.unpack('<IHHHHHIIIHH', f.read(30))
            if header[0] != 0x04034b50:
                raise IOError(f'corrupt local header for {m.filename}')
            f.seek(header[9] + header[10], 1)
            data = f.read(m.compress_size)
            if m.compress_type == zipfile.ZIP_DEFLATED:
                data = zlib.decompress(data, -15)
            out = os.path.join(dataroot, m.filename[len(strip):])
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, 'wb') as o:
                o.write(data)


def download_graphs(dataroot='./data/TCGA_GBMLGG'):
    """
    Fetch only the cell graphs (`all_st_cpc`), the split files that are not
    on disk yet and the clinical / genomic tables from the 14 GB archive.
    """
    url = DRIVE_URL.format(id=DATA_FILES['gbmlgg_data']['id'])
    size = remote_size(url)
    print("Reading the archive index ...")
    members = sorted(zipfile.ZipFile(_RemoteTail(url, size)).infolist(),
                     key=lambda m: m.header_offset)
    ends = {m.header_offset: (members[i + 1].header_offset
                              if i + 1 < len(members) else None)
            for i, m in enumerate(members)}

    splits_dir = os.path.join(dataroot, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    have = set(os.listdir(splits_dir))
    tmp = os.path.join(dataroot, '_range_tmp')
    os.makedirs(tmp, exist_ok=True)

    singles = [m for m in members if not m.is_dir() and (
        m.filename.count('/') == 1
        or (m.filename.startswith('TCGA_GBMLGG/splits/')
            and os.path.basename(m.filename) not in have))]
    for m in singles:
        if ends[m.header_offset] is None:
            continue
        blob = os.path.join(tmp, os.path.basename(m.filename) + '.range')
        print(f"Fetching {m.filename} ...")
        download_range(url, blob, m.header_offset, ends[m.header_offset])
        _extract_from_range(blob, m.header_offset, [m], dataroot)
        os.remove(blob)

    graphs = [m for m in members
              if m.filename.startswith('TCGA_GBMLGG/all_st_cpc/')]
    graph_dir = os.path.join(dataroot, 'all_st_cpc')
    n_files = sum(not m.is_dir() for m in graphs)
    if os.path.isdir(graph_dir) and len(os.listdir(graph_dir)) >= n_files:
        print("Cell graphs already on disk.")
    else:
        start = graphs[0].header_offset
        end = ends[graphs[-1].header_offset]
        blob = os.path.join(tmp, 'all_st_cpc.range')
        print(f"Fetching {n_files} cell graphs ({(end - start) / 1e6:.0f} MB) ...")
        download_range(url, blob, start, end)
        _extract_from_range(blob, start, graphs, dataroot)
        os.remove(blob)
    if not os.listdir(tmp):
        os.rmdir(tmp)
    print("Cell graphs, splits and clinical tables are ready.")


if __name__ == '__main__':
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    targets = list(DATA_FILES.keys()) + ['gbmlgg_graphs', 'all']
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='gbmlgg_splits',
                        choices=targets)
    # kept for backward compatibility with the documented command
    parser.add_argument('--dataset', type=str, default='TCGA_GBMLGG')
    args = parser.parse_args()

    if args.target == 'all':
        for k in DATA_FILES.keys():
            download_and_extract(k)
    elif args.target == 'gbmlgg_graphs':
        download_graphs()
    else:
        download_and_extract(args.target)
