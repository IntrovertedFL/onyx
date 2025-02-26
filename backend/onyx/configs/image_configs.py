"""
Centralized image-related configuration settings.
"""
import os

# Image processing configurations
ENABLE_IMAGE_EXTRACTION = (
    os.environ.get("ENABLE_IMAGE_EXTRACTION", "true").lower() == "true"
)
ENABLE_INDEXING_TIME_IMAGE_ANALYSIS = not (
    os.environ.get("DISABLE_INDEXING_TIME_IMAGE_ANALYSIS", "false").lower() == "true"
)
ENABLE_SEARCH_TIME_IMAGE_ANALYSIS = not (
    os.environ.get("DISABLE_SEARCH_TIME_IMAGE_ANALYSIS", "false").lower() == "true"
)
IMAGE_ANALYSIS_MAX_SIZE_MB = int(os.environ.get("IMAGE_ANALYSIS_MAX_SIZE_MB", "20"))

# This is for backward compatibility - will be removed in future versions
# Use ENABLE_INDEXING_TIME_IMAGE_ANALYSIS instead
DISABLE_INDEXING_TIME_IMAGE_ANALYSIS = not ENABLE_INDEXING_TIME_IMAGE_ANALYSIS
DISABLE_SEARCH_TIME_IMAGE_ANALYSIS = not ENABLE_SEARCH_TIME_IMAGE_ANALYSIS
