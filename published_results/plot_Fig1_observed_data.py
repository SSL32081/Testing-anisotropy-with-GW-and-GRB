#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("../data/")


# Observed GW skyloc maps
observed_map = np.load(DATA_DIR / 'GWTC-4_mixed_combined_skymap.npy')