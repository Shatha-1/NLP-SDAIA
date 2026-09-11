#!/usr/bin/env bash
set -euo pipefail
# Lab 7: provided-style wrapper. Requires `hey` installed separately.
#
# `hey` (and the Go toolchain needed to build it) were not available in the
# development environment this project was built in -- use
# `python scripts/load_test.py` instead, which reproduces the same load
# pattern (16 concurrent workers, 60s, same endpoint/payload) and reports the
# same p50/p99/error-count summary. See BENCHMARKS.md "HTTP load test" for the
# measured results either tool would report.
hey -z 60s -c 16 -m POST -H 'Content-Type: application/json' \
  -d '{"text":"الخدمة ممتازة ولكن التأخير طويل"}' \
  http://127.0.0.1:8000/v1/classify
