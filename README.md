# Steps to run the script:

### Team

Group number: 6

- Pandelis Laurens Symeonidis
- Elias Miguel Leal Spohr
- Lukas Gabriel Sekinger
- Daniel Hess

### Setting up Environment

This section shows how to create a virtual environment and how to install dependencies using the `requirements.txt`



1. Create virtual env

```bash
python3 -m venv venv/
```

2. Open virtual env

```bash
# For Unix
source venv/Scripts/activate
```

```bash
# For Windows
venv/Scripts/activate
```

3. Install required dependencies

```bash
python -m pip install -r requirements.txt
```

you can check if dependencies were installed by running next
command,it should print list with installed dependencies

```bash
pip list
```

5. Close virtual env

```bash
deactivate
```

## Quick Quide 

1. Activate virtual env

```bash
source venv/Scripts/activate
# or 
venv/Scripts/activate
```

2. Running the main script

```bash
python3 src/main.py [OPTIONS]
```

3. Close virtual env

```bash
deactivate
```

#### Example Commands

**Example 1: Running K-IBL with default preprocessing:**

```bash
python3 src/main.py \
  --model k_ibl \
  --dataset pen-based \
  --k 5 \
  --distance-metric cosine \
  --voting-strategy borda \
  --retention-strategy different_class_retention
```

**Example 2: Running Feature-Weighted K-IBL with custom preprocessing:**

```bash
python3 src/main.py \
  --model fw_k_ibl \
  --dataset pen-based \
  --feature-weighting-strategy relieff \
  --normalization mean_normalize \
  --encoding one_hot_encode \
  --missing-numeric-strategy median \
  --missing-categorical-strategy mode \
  --k 5 \
  --distance-metric cosine \
  --voting-strategy borda \
  --retention-strategy different_class_retention
```
**Example 2.1 (windows): Running Feature-Weighted K-IBL with custom preprocessing:**

```bash
python3 src/main.py  --model fw_k_ibl  --dataset pen-based  --feature-weighting-strategy relieff  --normalization mean_normalize  --encoding one_hot_encode  --missing-numeric-strategy median  --missing-categorical-strategy mode  --k 5  --distance-metric cosine  --voting-strategy borda  --retention-strategy different_class_retention
```

**Example 3: Running SVM:**

```bash
python3 src/main.py \
  --model svm \
  --dataset pen-based \
  --svm-kernel rbf \
  --normalization mean_normalize \
  --encoding label_encode
```
**Example 4 (windows): Running Feature-Weighted K-IBL with custom preprocessing:**

```bash
python3 src/main.py  --model ir_k_ibl  --dataset pen-based  --instance-reduction-strategy CNN  --normalization mean_normalize  --encoding one_hot_encode  --missing-numeric-strategy median  --missing-categorical-strategy mode  --k 5  --distance-metric cosine  --voting-strategy borda  --retention-strategy different_class_retention
```
**Example 5 (windows): Running Statistical Analysis (compatible with any number of files, and all types of configs):**
```bash
python3 src/statistical_analysis.py filename_1 filename_2 filename_k --metric f1_macro --export-csv output_filename --savefig filename.png
```

**Example 6 (windows): Computing plots from short_list data (Uncomment `grouped_diagrams.py line 9` to switch the dataset):**
```bash
python src/grouped_diagrams.py
```


#### Available Options

##### Model Selection

- `--model`: Model type to use
  - Options: `k_ibl`, `fw_k_ibl`, `ir_k_ibl`, `svm`
  - Default: `k_ibl`
  - **Note**: Model-specific arguments are required based on the selected model (see requirements below)

##### Dataset

- `--dataset`: Dataset name
  - Default: `pen-based`

##### Preprocessing Options

- `--normalization`: Normalization strategy

  - Options: `mean_normalize`, `standardize`, `unit_vector`, `minmax_scaling`
  - Default: `mean_normalize`

- `--encoding`: Encoding strategy

  - Options: `label_encode`, `one_hot_encode`
  - Default: `label_encode`

- `--missing-numeric-strategy`: Missing values strategy for numeric features

  - Options: `mean`, `median`, `zero`, `drop`, `model`
  - Default: `mean`

- `--missing-categorical-strategy`: Missing values strategy for categorical features
  - Options: `mode`, `constant`, `drop`
  - Default: `mode`

##### K-IBL Model Options (Required for `k_ibl`, `fw_k_ibl`, `ir_k_ibl`)

- `--k`: Number of nearest neighbors

  - Options: `3`, `5`, `7`
  - Required for all k-IBL variants

- `--distance-metric`: Distance metric to use

  - Options: `euclidean`, `cosine`, `heom`
  - Required for all k-IBL variants

- `--voting-strategy`: Voting strategy

  - Options: `modified_plurality`, `borda`
  - Required for all k-IBL variants

- `--retention-strategy`: Retention policy
  - Options: `never_retain`, `always_retain`, `different_class_retention`, `DD_retention`
  - Required for all k-IBL variants

##### Feature-Weighted K-IBL Options (Required for `fw_k_ibl`)

- `--feature-weighting-strategy`: Feature weighting strategy
  - Options: `relieff`, `information_gain`
  - Required when `--model` is `fw_k_ibl`

##### Instance Reduction K-IBL Options (Required for `ir_k_ibl`)

- `--instance-reduction-strategy`: Instance reduction type
  - Options: `ibl3`
  - Required when `--model` is `ir_k_ibl`

##### SVM Options (Required for `svm`)

- `--svm-kernel`: SVM kernel type
  - Options: `rbf`, `poly`
  - Required when `--model` is `svm`

#### Model-Specific Requirements

Different models have different argument requirements:

- **K-IBL variants** (`k_ibl`, `fw_k_ibl`, `ir_k_ibl`): Require `--k`, `--distance-metric`, `--voting-strategy`, and `--retention-strategy`
- **Feature-Weighted K-IBL** (`fw_k_ibl`): Additionally requires `--feature-weighting-strategy`
- **Instance Reduction K-IBL** (`ir_k_ibl`): Additionally requires `--instance-reduction-strategy`
- **SVM** (`svm`): Requires `--svm-kernel`

#### Viewing Help

To see all available options and their descriptions:

```bash
python3 src/main.py --help
```

#### Output

The script outputs results to a CSV file (`idfk.csv` in the project root by default) containing performance metrics for each fold including:

- Accuracy, precision, recall, and F1 scores (macro and weighted averages)
- Confusion matrices (as JSON)
- Training and prediction times
- Dataset and configuration information

# How to Run the Tests

To run all tests and see coverage in your terminal, execute the following command from the project root:

```sh
pytest
```

This will automatically discover and run all tests in the `tests/` directory. Coverage information will be displayed in the terminal, and tests will run in parallel if possible.

If you want to run a specific test file, use:

```sh
pytest tests/test_distance_measures.py
```

# BEST KIBL

cozine
k=7
voting=borda
retention=diff_class_retention

normalize: mean normalize
missing-value-num = median
missing value cat = mode