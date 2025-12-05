import sys
from unittest.mock import MagicMock

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
