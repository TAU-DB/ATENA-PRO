"""
This file contains all data with regard to columns in the netflix schema
"""

KEYS = ['app_id', 'name', 'category', 'rating', 'reviews', 'app_size_kb', 'installs', 'type', 'price', 'content_rating', 'last_updated', 'min_android_ver']
KEYS_ANALYST_STR = KEYS
assert len(KEYS) == len(KEYS_ANALYST_STR)
FILTER_COLS = KEYS  # Note: changing this from KEYS require to change other occurrences of KEYS in the codebase
GROUP_COLS = KEYS  # Note: changing this from KEYS require to change other occurrences of KEYS in the codebase

NUMERIC_KEYS = {'app_id', 'rating', 'reviews', 'app_size_kb', 'installs', 'price'}
AGG_KEYS = ['app_id', 'rating', 'reviews', 'app_size_kb', 'installs', 'price']
AGG_KEYS_ANALYST_STR = AGG_KEYS
DONT_FILTER_FIELDS = {'app_id'}

