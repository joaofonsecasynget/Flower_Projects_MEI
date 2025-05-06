"""
Utilitários para geração de relatórios e visualizações no cliente federado.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import torch
import time
import shap
import re
from datetime import datetime

from .report_helpers import (
    extract_feature_name, format_value, get_interpretation,
    generate_evolution_plots,
    generate_html_report,
    save_artifacts,
    generate_explainability, generate_explainability_formatado
)

# Configuração do logger
logger = logging.getLogger()
