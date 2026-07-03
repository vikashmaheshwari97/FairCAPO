"""Small process-wide runtime safeguards for FairCAPO command-line scripts.

Python imports ``sitecustomize`` automatically when the repository root is on
``PYTHONPATH`` (all HPC launchers export ``PYTHONPATH=.``).  Portfolio CSV rows
may contain serialized prompts and few-shot examples larger than the standard
128 KiB csv field limit, so raise the limit once for every project script.
"""

import csv

csv.field_size_limit(10 * 1024 * 1024)
