# Project Overview
A script that collects grant data from URLs and outputs standardized JSONL records.

# Installation

These are **project-specific Python dependencies** required to run this script (they are not part of the Python standard library).

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install httpx beautifulsoup4 lxml python-dateutil dateparser
```

# Usage
Single URL:
```
python main.py --url https://example.org/grant
```
Note: `test_url.txt` is a test file with a single URL.

Multiple URLs:
```
python main.py --input test_urls.txt
```
Note: `test_urls.txt` is a test file with multiple URLs.

# Outputs
Default output (timestamped):
```
output_YYYYMMDD_HHMMSS.jsonl
```

Run report (missing fields and failed URLs):
```
report for output output_YYYYMMDD_HHMMSS.json
```