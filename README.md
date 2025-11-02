# Steps to run the script:

## Team

Group number: 6

- Pandelis Laurens Symeonidis
- Elias Miguel Leal Spohr
- Lukas Gabriel Sekinger
- Daniel Hess

## Running script for the first time

This section shows how to create a virtual environment and how to install dependencies using the `requirements.txt`

1. Open folder in terminal

```bash
cd <root_folder_of_project>/
```

2. Create virtual env

```bash
# For Unix
python3 -m venv venv/
```

```bash
# For Windows
python -m venv venv/
```

3. Open virtual env

```bash
# For Unix
source venv/bin/activate
```

```bash
# For Windows
venv/Scripts/activate
```

4. Install required dependencies

```bash
# For Unix
pip install -r requirements.txt
```

```bash
# For Windows
python -m pip install -r requirements.txt
```

5. Close virtual env

```bash
deactivate
```

you can check if dependencies were installed by running next
command,it should print list with installed dependencies

```bash
pip list
```

## Execute scripts

1. Open virtual env

```bash
source venv/bin/activate
```

2. Running the script.

All implemented models are available from the `src/main.py` script. You can configure the script it by passing a set of arguments as explained in the section below (Script configuration). For instance, executing the k-IBL model would look like this:

```bash
python3 src/main.py --model k_ibl --dataset pen-based --k 5 --distance-metric cosine --voting-strategy borda --retention-strategy different_class_retention --out_filename pen-based-k_ibl.csv
```

\*Replace `python3` with `python` if running on Windows.

1. Close virtual env

```bash
deactivate
```

## Script configuration

All implemented models are available through the `src/main.py` script. The script accepts a set of arguments. To quickly see what arguments are required you can run:

```bash
python3 src/main.py --help
```

The complete list of parameters are:

### Core Parameters (Always Available)

- **`--dataset`**: The name of the dataset to use (e.g., `pen-based`, `adult`)

- **`--model`**: The machine learning model to run. This is the primary parameter that determines which other parameters will be required.

  - Options: `k_ibl`, `fw_k_ibl`, `ir_k_ibl`, `svm`
  - **Important**: The model you choose determines which additional parameters are required (see model-specific requirements below)

- **`--out_filename`**: The output filename for the CSV results file (e.g., `results.csv`)
  - Required: Yes
  - The file will be saved in the `src/results/` directory

### K-IBL Parameters (Required for `k_ibl`, `fw_k_ibl`, and `ir_k_ibl` models)

These four parameters are mandatory when running any k-IBL variant:

- **`--k`**: Number of nearest neighbors to consider for classification

  - Options: `3`, `5`, `7`

- **`--distance-metric`**: The distance metric used to measure similarity between instances

  - Options: `euclidean`, `cosine`, `heom`

- **`--voting-strategy`**: How the k nearest neighbors vote to determine the final classification
  - Options: `modified_plurality`, `borda`
- **`--retention-strategy`**: Policy for retaining correctly classified instances in memory
  - Options: `never_retain`, `always_retain`, `different_class_retention`, `DD_retention`

### Feature-Weighted K-IBL Parameters (Required only for `fw_k_ibl`)

- **`--feature-weighting-strategy`**: Method to calculate feature importance/weights
  - Options: `relieff`, `information_gain`

### Instance Reduction K-IBL Parameters (Required only for `ir_k_ibl`)

- **`--instance-reduction-strategy`**: Method to reduce the training set size before classification
  - Options: `IBL3`, `CNN`, `enn`

### SVM Parameters (Required only for `svm` model)

When using the SVM model, all four of these parameters are mandatory:

- **`--svm-kernel`**: The kernel function used by the SVM
  - Options: `rbf`, `poly`
- **`--C`**: Regularization parameter that controls the trade-off between margin maximization and classification error

  - Options: `0.01`, `0.1`, `1`, `10`

- **`--gamma`**: Kernel coefficient for RBF and polynomial kernels

  - Options: `0.001`, `0.01`, `0.1`

- **`--degree`**: Degree of the polynomial kernel (only used when `--svm-kernel poly`)
  - Options: `2`, `3`

```bash
python3 src/main.py --model k_ibl --dataset pen-based --k 5 --distance-metric cosine --voting-strategy borda --retention-strategy different_class_retention --out_filename pen-based-k_ibl.csv
```

## Example script runs

**Important** If running on Windows make sure to use the `python` command.

**Example 1: Running k-IBL:**

```bash
python3 src/main.py --model k_ibl --dataset pen-based --k 5 --distance-metric cosine --voting-strategy borda --retention-strategy different_class_retention --out_filename pen-based-k_ibl.csv
```

**Example 2: Running Feature-Weighted K-IBL:**

```bash
python3 src/main.py --model fw_k_ibl --dataset pen-based --feature-weighting-strategy relieff --k 5 --distance-metric cosine --voting-strategy borda --retention-strategy different_class_retention --out_filename pen-based-fw_k_ibl.csv
```

**Example 3: Running SVM:**

```bash
python3 src/main.py --model svm --dataset pen-based --svm-kernel rbf --C 1 --gamma 0.01 --degree 2 --out_filename pen-based-svm.csv
```

**Example 4: Running k-IBL with Instance Reduction:**

```bash
python3 src/main.py  --model ir_k_ibl  --dataset pen-based  --instance-reduction-strategy CNN --k 5  --distance-metric cosine  --voting-strategy borda  --retention-strategy different_class_retention --out_filename pen-based-ir_k_ibl.csv
```

**Example 5: Running Statistical Analysis (compatible with any number of files, and all types of configs):**

```bash
python src/statistical_analysis.py filename_1 filename_2 filename_k --metric f1_macro --export-csv output_filename --savefig filename.png
python src/statistical_analysis.py "src/results/pen-based-ir_k_ibl copy.csv" src/results/pen-besed-k_ibl_best.csv --metric f1_macro --export-csv adult_CD --savefig adult_cd.png
```

**Example 6: Creating plots from data (Uncomment `grouped_diagrams.py line 9` to switch the dataset):**

```bash
python src/grouped_diagrams_ibl.py
python src/grouped_diagrams_svm.py
```

## Output

The script outputs results to a CSV file in the `src/results/` directory. The filename is specified using the `--out_filename` parameter. The CSV file contains performance metrics for each fold including:

- Accuracy, precision, recall, and F1 scores (macro and weighted averages)
- Confusion matrices (as JSON)
- Training and prediction times
- Dataset and configuration information

## How to Run the Unit Tests

To run all unittests and see coverage in your terminal, execute the following command from the project root:

```sh
pytest
```

This will automatically discover and run all tests in the `tests/` directory. Coverage information will be displayed in the terminal, and tests will run in parallel if possible.

If you want to run a specific test file, use:

```sh
pytest tests/test_distance_measures.py
```
