# Coordinators Horizon Europe

This repository contains tools to find and analyze coordinators for Horizon Europe projects based on text descriptions of future calls.

## Overview

The project includes Python scripts designed to download project data from the CORDIS database and use natural language processing (NLP) to find the most relevant Horizon Europe projects and their coordinators based on user-provided keywords or call descriptions.

## Files

- **`find_coordinator.py`**: The main script. It downloads the CORDIS Horizon Europe project and organization datasets, uses TF-IDF vectorization and cosine similarity to find projects matching a given query, and outputs the details of the coordinating organizations.
- **`get_urls.py`**: A utility script that fetches the latest CSV download URLs for Horizon Europe from the EU Open Data portal and saves them to `urls.txt`.
- **`requirements.txt`**: Contains the Python dependencies required to run the scripts.

## Installation

Ensure you have Python installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

You can run the coordinator finder directly from the command line:

```bash
python find_coordinator.py "your call description or keywords here"
```

### Options

- `query`: Text description or keywords of the future call. If omitted, the script will prompt you for input.
- `--top`: Number of top matching projects to return (default: 10).
- `--csv`: Optional filename to save the results as a CSV file.

### Example

```bash
python find_coordinator.py "Artificial Intelligence in healthcare" --top 5 --csv results.csv
```

This will find the top 5 projects related to "Artificial Intelligence in healthcare", print the coordinating organizations to the console, and save the detailed results to `results.csv`.
