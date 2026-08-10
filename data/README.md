# Data

The BreakHis dataset is not committed to this repo (it's several GB).

## How to get it
1. Download from: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
   (requires filling out a short access form)
2. Alternatively, search Kaggle for "BreakHis" — several mirrors exist there
   and are quicker to download, but verify the license/terms before using.

## Expected structure after download
```
data/
└── raw/
    └── BreaKHis_v1/
        └── histology_slides/
            └── breast/
                ├── benign/
                └── malignant/
```

Place the extracted dataset under `data/raw/`. This folder is gitignored,
so nothing here gets committed — every collaborator needs to download it
separately.
