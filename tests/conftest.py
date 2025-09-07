
import sys
from unittest.mock import MagicMock
import pytest

@pytest.fixture(autouse=True)
def mock_heavy_imports(monkeypatch):
    """
    Mocks modules that cause import errors or are not needed for tests.
    'autouse=True' ensures this runs before any tests.
    """
    # Mock streamlit and its internal modules
    mock_streamlit = MagicMock()
    sys.modules['streamlit'] = mock_streamlit
    sys.modules['streamlit.runtime.caching.cache_utils'] = MagicMock()

    # Mock tensorflow to prevent its import
    mock_tensorflow = MagicMock()
    sys.modules['tensorflow'] = mock_tensorflow
    sys.modules['tensorflow.keras.models'] = MagicMock()
    
    # Mock shap
    sys.modules['shap'] = MagicMock()
    
    # Mock plotly
    sys.modules['plotly'] = MagicMock()
    sys.modules['plotly.express'] = MagicMock()

