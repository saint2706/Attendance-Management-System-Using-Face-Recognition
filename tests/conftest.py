import sys
from unittest.mock import MagicMock

import pytest

# Attempt to import tensorflow/deepface. If they fail, mock them.
try:
    import deepface  # noqa: F401
    import tensorflow  # noqa: F401
except ImportError:
    # Mock tensorflow
    tf_mock = MagicMock()
    sys.modules["tensorflow"] = tf_mock
    sys.modules["tensorflow.python"] = tf_mock
    sys.modules["tensorflow.python.pywrap_tensorflow"] = tf_mock

    # Mock deepface
    df_mock = MagicMock()
    sys.modules["deepface"] = df_mock
    sys.modules["deepface.DeepFace"] = df_mock


@pytest.fixture(scope="session", autouse=True)
def close_database_connections():
    """Ensure all database connections are properly closed after tests.

    This fixture runs at the end of the test session to prevent the
    'database is being accessed by other users' error during teardown.
    """
    yield
    # Close all database connections after tests complete
    # Import inside the fixture to avoid issues if Django is not configured
    try:
        from django.db import connections

        for conn in connections.all():
            conn.close()
    except Exception:
        # If Django is not configured or connections can't be closed, ignore
        pass
