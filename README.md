# NoteGuard — Banknote Authenticity Classifier

NoteGuard is a full-stack Machine Learning web application designed to classify banknotes as **GENUINE** or **FORGED**. It supports dual verification modes:
1. **Image Upload Analysis**: Extracts four discrete wavelet transform features in-memory from a banknote photograph or scan (`variance`, `skewness`, `curtosis`, `entropy`).
2. **Manual Feature Entry**: Allows manual evaluation of numerical feature parameters, equipped with 1-click test specimen loaders from the UCI dataset.

---

## Architecture & Tech Stack

- **Backend**: Python 3.10+, Flask 3.x
- **Machine Learning**: scikit-learn (Decision Tree Classifier, Random Forest Classifier), Joblib
- **Image Processing & Wavelets**: Pillow, NumPy, SciPy, PyWavelets (`pywt`)
- **Frontend**: Responsive HTML5, CSS3 (Dark Navy theme, `#121622` to `#1e2638`), Vanilla JavaScript (ES6+)
- **Visualizations**: Chart.js (CDN), Font Awesome 6 (CDN), Google Fonts (IBM Plex Sans)

---

## Dataset & Label Semantics

- **Source**: [UCI Machine Learning Repository — Banknote Authentication Dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication) (ID: 267)
- **Specimens**: 1,372 records (762 Genuine, 610 Forged).
- **Target Variable**: `class`
  - `0` = **Genuine** (Negative class, 55.4%)
  - `1` = **Forged** (Positive class, 44.6%)
- **Data Validation Summary**:
  - Missing values: `0` across all columns.
  - Duplicates: `24` duplicate rows (preserved as identical wavelet feature vectors can arise from identical banknote batches).
  - Train/Test Split: Stratified 80/20 split (`random_state=42`), 1,097 training records, 275 test records.

---

## Machine Learning Results & Metrics

Both models were trained using 5-fold Stratified Cross-Validation with `GridSearchCV` hyperparameter tuning on the training set, and evaluated on the held-out 20% test set (275 specimens). **Forged (`class = 1`) is treated as the positive class.**

| Metric | Decision Tree Classifier | Random Forest Classifier (Best Model) |
| :--- | :--- | :--- |
| **Accuracy** | **99.27%** (273 / 275) | **99.64%** (274 / 275) |
| **Precision** | **98.39%** | **99.19%** |
| **Recall** | **100.00%** | **100.00%** |
| **F1-Score** | **99.19%** | **99.59%** |
| **Best Parameters** | `criterion='entropy'`, `max_depth=8`, `min_samples_split=2` | `n_estimators=150`, `max_depth=8`, `min_samples_split=2`, `max_features='sqrt'` |

### Confusion Matrices (Test Set, N = 275)

#### Decision Tree
```
                    Predicted Genuine (0)    Predicted Forged (1)
Actual Genuine (0)           151 (TN)                 2 (FP)
Actual Forged (1)              0 (FN)               122 (TP)
```

#### Random Forest (Winner)
```
                    Predicted Genuine (0)    Predicted Forged (1)
Actual Genuine (0)           152 (TN)                 1 (FP)
Actual Forged (1)              0 (FN)               122 (TP)
```

### Feature Importance

| Feature | Description | Decision Tree Weight | Random Forest Weight |
| :--- | :--- | :--- | :--- |
| `variance` | Variance of wavelet transformed image | **53.81%** | **55.80%** |
| `skewness` | Skewness of wavelet transformed image | **29.22%** | **23.47%** |
| `curtosis` | Excess curtosis of wavelet transformed image | **14.70%** | **15.06%** |
| `entropy` | Continuous Shannon entropy of image | **2.28%** | **5.67%** |

---

## Image Feature Extraction Decisions & Methodology

The image extraction pipeline in `feature_extraction.py` processes raw image bytes strictly in-memory (uploads are never written to disk):

1. **Format & Integrity Check**: Pillow decodes and calls `.verify()` to reject corrupt payloads or disguised text files. Supported formats: PNG, JPG, JPEG, GIF, BMP. For multi-frame GIF files, the first frame is selected.
2. **Grayscale & Normalization**: The image is converted to single-channel grayscale (`"L"`), resized to `(400, 400)` with `Image.Resampling.LANCZOS` (the exact native resolution of the UCI dataset), and mapped to float64 pixel intensities in `[0.0, 1.0]`.
3. **Discrete Wavelet Transform (`cA` vs `cH`)**:
   - Single-level 2D discrete wavelet transform is computed using the Haar wavelet (`pywt.dwt2(norm_arr, "db1")`).
   - **Decision: `cA` (Approximation Coefficients) was selected.**
   - *Rationale*: Detailed empirical testing demonstrated that horizontal detail coefficients (`cH`) capture near-zero high-frequency edge gradients (`var ≈ 0.003`, kurtosis spikes to `+6.0`), whereas approximation coefficients (`cA`) preserve the macroeconomic banknote luminance distribution and textural depth, yielding feature values (`var ≈ 0.15–0.35`, `skew ≈ -0.9 to +0.8`, `kurt ≈ -1.6 to +0.5`) that match the UCI dataset's training range.
4. **Kurtosis Definition (Fisher vs Pearson)**:
   - **Decision: Fisher's Excess Kurtosis (`fisher=True`) was chosen.**
   - *Rationale*: Pearson's kurtosis is strictly non-negative (normal distribution = 3.0, minimum possible value = 1.0). In contrast, the UCI dataset exhibits negative curtosis values spanning from `-5.26` to `+17.93` (mean `+1.41`). Fisher's definition centers the normal distribution at `0.0` and naturally accommodates negative excess kurtosis.
5. **Entropy Formulation**:
   - Discrete Shannon entropy from a 256-bin normalized histogram is positive ($H \in [0, 8]$ bits). However, the UCI continuous wavelet dataset features negative entropy values ($[-8.55, +2.14]$, mean $-1.21$).
   - To align with continuous wavelet transform methodology, continuous differential entropy is used:
     $$\hat{H}_{\text{diff}} = -\sum p_i \log_2(p_i) + \log_2(\Delta x)$$
     where $\Delta x = \frac{\max(cA) - \min(cA)}{256}$. Because $\Delta x \approx \frac{1}{128}$, $\log_2(\Delta x) \approx -7.0$, which shifts typical discrete entropy ($\approx 5.8$) right into the dataset's empirical center ($\approx -1.2$).
6. **Out-of-Distribution Sanity Check**:
   - Each extracted feature is compared against the training dataset bounds with a 5% tolerance margin:
     - `variance`: `[-7.73, 7.51]`
     - `skewness`: `[-15.11, 14.29]`
     - `curtosis`: `[-6.42, 19.09]`
     - `entropy`: `[-9.08, 2.67]`
   - If any feature falls outside, `out_of_distribution: true` is flagged and a yellow warning is displayed on the UI.

---

## Folder Structure

```
NoteGuard — Banknote Authenticity Classifier/
├── app.py                      # Flask application and REST API endpoints
├── train_model.py              # ML training, GridSearchCV tuning, and serialization
├── feature_extraction.py       # Wavelet feature extraction pipeline
├── requirements.txt            # Pinned dependency requirements
├── README.md                   # Complete documentation
├── data/
│   └── banknote.csv            # UCI Banknote Authentication dataset
├── models/
│   ├── decision_tree.joblib    # Trained Decision Tree classifier
│   ├── random_forest.joblib    # Trained Random Forest classifier
│   └── metrics.json            # Model performance, CM, importances, training stats
├── templates/
│   ├── index.html              # Prediction interface (Upload & Manual tabs)
│   └── dashboard.html          # Evaluation dashboard with Chart.js charts
├── static/
│   ├── css/
│   │   └── style.css           # Dark navy theme stylesheet
│   └── js/
│       ├── main.js             # Prediction page frontend logic
│       └── dashboard.js        # Evaluation dashboard Chart.js controller
├── test_feature_extraction.py  # Unit tests for wavelet feature extraction
└── test_app_endpoints.py       # Integration tests for all Flask API routes
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Git (optional)

### 1. Clone or Open Workspace
```bash
cd "c:\Users\mabis\OneDrive\Documents\Projects\Banknote Authenticity Classifier"
```

### 2. Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS / Linux (bash/zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train Models & Generate Metrics
```bash
python train_model.py
```
This loads `data/banknote.csv` (or downloads it automatically from the UCI repository), executes 5-fold cross-validation with grid search, evaluates the models, and generates `models/decision_tree.joblib`, `models/random_forest.joblib`, and `models/metrics.json`.

### 5. Run Self-Tests & Integration Tests
```bash
# Run feature extraction unit tests
python test_feature_extraction.py

# Run Flask endpoint integration tests
python test_app_endpoints.py
```

### 6. Start the NoteGuard Web Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```
- **Prediction Interface**: `http://127.0.0.1:5000/`
- **Evaluation Dashboard**: `http://127.0.0.1:5000/dashboard`

---

## API Documentation

### 1. Manual Prediction
- **Endpoint**: `POST /api/predict`
- **Content-Type**: `application/json`
- **Request Body**:
```json
{
  "variance": 3.6216,
  "skewness": 8.6661,
  "curtosis": -2.8073,
  "entropy": -0.44699,
  "model": "random_forest"
}
```
- **Response** (`200 OK`):
```json
{
  "class_index": 0,
  "confidence": 100.0,
  "model_key": "random_forest",
  "model_used": "Random Forest",
  "prediction": "Genuine",
  "probabilities": {
    "Forged": 0.0,
    "Genuine": 100.0
  }
}
```

#### Example `curl` command:
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"variance": 3.6216, "skewness": 8.6661, "curtosis": -2.8073, "entropy": -0.44699, "model": "random_forest"}'
```

---

### 2. Image Upload Prediction
- **Endpoint**: `POST /api/predict-image`
- **Content-Type**: `multipart/form-data`
- **Form Fields**:
  - `image`: Binary image file (PNG, JPG, JPEG, GIF, BMP, max 16MB)
  - `model`: `"random_forest"` or `"decision_tree"` (optional, defaults to best model)
- **Response** (`200 OK`):
```json
{
  "class_index": 1,
  "confidence": 84.17,
  "extracted_features": {
    "variance": 0.1431,
    "skewness": -0.0806,
    "curtosis": -1.6087,
    "entropy": -5.9939
  },
  "metadata": {
    "coefficient_type": "cA (Approximation)",
    "kurtosis_type": "Fisher (Excess)",
    "original_format": "PNG",
    "original_mode": "L",
    "original_size": [400, 400],
    "processed_size": [400, 400],
    "wavelet_used": "db1"
  },
  "model_key": "random_forest",
  "model_used": "Random Forest",
  "out_of_distribution": false,
  "out_of_range_features": [],
  "prediction": "Forged",
  "probabilities": {
    "Forged": 84.17,
    "Genuine": 15.83
  },
  "warning": null
}
```

#### Example `curl` command:
```bash
curl -X POST http://127.0.0.1:5000/api/predict-image \
  -F "image=@sample_note.png" \
  -F "model=random_forest"
```

---

### 3. Model Metrics & Ranges
- **Endpoint**: `GET /api/metrics`
- **Response**: Full `metrics.json` payload containing accuracy, confusion matrices, feature importances, dataset balance, and training ranges.

### 4. Demo Samples
- **Endpoint**: `GET /api/samples`
- **Response**: Real genuine and forged rows from the UCI dataset used by the UI's quick-fill buttons.

---

## UI Screenshots

### 1. Upload Currency Note Interface
*(Dark navy layout featuring file drag & drop, client-side validation, live image preview, verdict badge, and extracted wavelet features)*

```
+------------------------------------------------------------------------------------+
|  [Shield] NoteGuard  Banknote Authenticity Classifier         [Analyze]  [Dashboard]|
+------------------------------------------------------------------------------------+
|                                                                                    |
|                   Verify Banknote Authenticity                                     |
|                   [Upload Image (Active)]   [Manual Entry]                         |
|                                                                                    |
|   +----------------------------------------------------------------------------+   |
|   | [Cloud] Upload Currency Note                  Model: [Random Forest (Best)]|   |
|   |                                                                            |   |
|   |  + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +   |   |
|   |  |   [Image Icon]  Drag & Drop banknote image here                     |   |   |
|   |  |   [Choose file]  specimen_50.png                                    |   |   |
|   |  + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +   |   |
|   |  (i) Supported formats: PNG, JPG, JPEG, GIF, BMP (Max size: 16MB)           |   |
|   |                                                                            |   |
|   |  [=========== Analyze Currency Note (Magnifier) ===========]               |   |
|   |                                                                            |   |
|   |  RESULT CARD:                                                              |   |
|   |  [Thumb]  GENUINE (Green Badge)        Confidence: 99.8% [==========]      |   |
|   |  Wavelet Features: Variance: 3.6216 | Skewness: 8.6661 | Curtosis: -2.8073 |   |
|   |  [Send these features to Manual Entry]                                     |   |
|   +----------------------------------------------------------------------------+   |
+------------------------------------------------------------------------------------+
```

### 2. Evaluation Dashboard
*(Comprehensive performance overview with side-by-side model spec cards, heatmap confusion matrices, and interactive Chart.js graphs)*

---

## Limitations & Educational Disclaimer

> [!WARNING]
> Please review the following technical constraints:
> 1. **Training Data Origin**: The machine learning models were trained on pre-extracted numeric wavelet features from the UCI Machine Learning repository, rather than end-to-end raw image convolutional networks.
> 2. **Image Feature Extraction Approximation**: Image-based prediction uses a mathematical approximation of the original digitization process (400x400 discrete wavelet transform). Photographs taken from consumer smartphones with perspective distortion, shadows, or varied sensor color matrices may differ from the original industrial camera scans.
> 3. **Out-of-Distribution Warning is a Heuristic**: The out-of-distribution detection checks whether extracted values fall within the min/max training boundaries ($\pm 5\%$). It serves as an advisory heuristic, not a mathematical certainty.
> 4. **Strictly Educational Purpose**: This software is an educational machine learning demonstration. **Do not use NoteGuard to verify real legal tender currency.**

