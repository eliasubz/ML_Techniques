# Project for Machine Learning Course at UB

## Installation

Before running or testing this project, developers must install all required dependencies. From the project root, run:

```sh
pip install -r requirement.txt
```

This will ensure all necessary Python packages are available for both development and testing.

### How to Run the Tests

To run all tests and see coverage in your terminal, execute the following command from the project root:

```sh
pytest
```

This will automatically discover and run all tests in the `tests/` directory. Coverage information will be displayed in the terminal, and tests will run in parallel if possible.

If you want to run a specific test file, use:

```sh
pytest tests/test_distance_measures.py
```

No coverage files will be written to disk by default (see `pytest.ini` for configuration).
