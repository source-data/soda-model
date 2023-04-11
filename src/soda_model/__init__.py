import sys
from dotenv import load_dotenv
import os

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "soda-model"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

load_dotenv()
CACHE = str(os.getenv("CACHE"))
DATA_FOLDER = str(os.getenv("DATA_FOLDER"))
RESULTS_FOLDER = str(os.getenv("RESULTS_FOLDER"))
MODELS_FOLDER = str(os.getenv("MODELS_FOLDER"))
TEST_FOLDER = str(os.getenv("TEST_FOLDER"))

# for path_folder in [CACHE, DATA_FOLDER, RESULTS_FOLDER, MODELS_FOLDER, TEST_FOLDER]:
#     if not os.path.exists(path_folder):
#         os.makedirs(path_folder)
