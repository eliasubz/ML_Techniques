# Steps to run the script:

### Team

- Pandelis Laurens Symeonidis
- Elias Miguel Leal Spohr
- Lukas Gabriel Sekinger
- Daniel Hess

### Running script for the first time

These sections show how to create virtual environment for
our script and how to install dependencies

1. Open folder in terminal

```bash
cd <root_folder_of_project>/
```

2. Create virtual env

```bash
python3 -m venv venv/
```

3. Open virtual env

```bash
source venv/bin/activate
```

4. Install required dependencies

```bash
pip install -r requirements.txt
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

## Execute scripts

1. Open virtual env

```bash
source venv/bin/activate
```

2. Running the script (add subpoints if you have more than one main).

```bash
   python3 main_name.py
```

3. Close virtual env

```bash
deactivate
```

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
