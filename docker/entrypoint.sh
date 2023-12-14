#!/bin/bash
set -eux

source /venv/bin/activate
streamlit run webui.py --server.port 8501