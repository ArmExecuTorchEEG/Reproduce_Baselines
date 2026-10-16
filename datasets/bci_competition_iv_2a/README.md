# BCI Competition IV Dataset 2a — local recordings

Open [recordings/](recordings/) to browse A01T.mat/A01E.mat through
A09T.mat/A09E.mat. T is Session 1, E is Session 2; each recording contains all
four imagery classes. Our model pipeline selects left/right only (144/session).

`recordings` is a RELATIVE symbolic link to
`../../mne_data/MNE-bnci-data/~bci/database/001-2014`. No raw recordings were
moved or copied. All 18 files remain in MOABB's existing ignored cache.
`src/data/download.py`, `inspect_dataset.py`, `preprocess.py` and the combined/CV
loaders continue to use repository-local `mne_data/`. This link is for navigation,
not an alternative MNE_DATA setting; do not change the loaders to point here.

After cloning, activate the pinned Python 3.12 environment and run from repo root:
```sh
python -m src.data.download --subjects all
python -m src.data.inspect_dataset
```
The link is expected to be broken until the local dataset is downloaded. macOS
and Linux preserve Git symlinks; on Windows enable Git symlink support, or browse
the documented cache path directly. A provider/cache-layout change may require
updating only this navigation link, not moving files. Git tracks the README and
link, never the raw data. Do not force-add recordings through either path.
