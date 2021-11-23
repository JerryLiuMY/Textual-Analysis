import os
from global_settings import LOG_PATH
from global_settings import CLEAN_PATH
from global_settings import RICH_PATH
from global_settings import WORD_PATH


# Create directories
if not os.path.isdir(LOG_PATH):
    os.mkdir(LOG_PATH)

if not os.path.isdir(CLEAN_PATH):
    os.mkdir(CLEAN_PATH)

if not os.path.isdir(RICH_PATH):
    os.mkdir(RICH_PATH)

if not os.path.isdir(WORD_PATH):
    os.mkdir(WORD_PATH)
