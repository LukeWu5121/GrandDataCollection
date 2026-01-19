# Project Overview
A script that collects grant data from URLs and outputs standardized JSONL records.

Installation
python -m venv .venv
source .venv/bin/activate
pip install httpx beautifulsoup4 lxml python-dateutil dateparser

Usage
# Single URL
python main.py --url https://example.org/grant
## Batch URLs
python main.py --input urls.txt

Outputs
# Default output (timestamped)
output_YYYYMMDD_HHMMSS.jsonl
# Run report (missing fields and failed URLs)
report for output output_YYYYMMDD_HHMMSS.json
