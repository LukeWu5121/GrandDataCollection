# Project Overview
A script that collects grant data from URLs and outputs standardized JSONL records.

# Description
Files in this folder:
- `Grand Data Collection Task Description`: Project File.
- `Log.txt`: Hand-Type Log to record changes.
- `main.py`: the scraper CLI that fetches pages, extracts fields, and writes outputs.
- `README.md`: usage and output documentation.
- `test_url.txt`: a test file with a single URL.
- `test_urls.txt`: a test file with multiple URLs.
- `test_urls_404.txt`: a robustness test file with timeouts, TLS errors, and HTTP errors.

Inputs and outputs:
- You provide one URL or a text file of URLs.
- The script writes two files per run: a JSONL output and a JSON report.

# Installation
Assumes Python 3.10+ is already installed.

These are **project-specific Python dependencies** required to run this script (they are not part of the Python standard library).

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install httpx beautifulsoup4 lxml python-dateutil dateparser
```

# Usage
<sub>Note: Depending on your system setup and current working directory, file paths may differ—adjust the script and input file paths accordingly.</sub>

Single URL:
```
python3 main.py --url https://example.org/grant
python3 main.py --input test_url.txt
```

Multiple URLs:
```
python3 main.py --input test_urls.txt
```

Robustness test (timeouts, TLS, HTTP errors):
```
python3 main.py --input test_urls_404.txt
```

# Outputs
Default output (timestamped):
```
output_YYYYMMDD_HHMMSS.jsonl
```
Each line is one unique URL record with fields like `agency`, `application_link`, `description`, `ein`, `published_date`, and `response_date`.

Run report (missing fields and failed URLs):
```
report for output_YYYYMMDD_HHMMSS.json
```
The report summarizes overall success/failure counts, missing fields, not-found URLs, and duplicates.