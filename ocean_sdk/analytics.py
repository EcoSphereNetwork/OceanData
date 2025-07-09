"""Analysis wrappers for OceanData."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from oceandata.core.ocandata_ai import OceanDataAI


def run_analysis(
    data: pd.DataFrame,
    source_type: str,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run built-in analysis pipelines for a given data source."""
    ai = OceanDataAI(config)
    return ai.analyze_data_source(data, source_type)
