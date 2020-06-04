
BASE_IMAGE_PATH="${SM_CHANNEL_TRAINING}/dark-forest-landscape.jpg"
RESULT_PREFIX="${SM_MODEL_DIR}/dream"

python deep_dream.py ${BASE_IMAGE_PATH} ${RESULT_PREFIX}

echo "Generated image $(ls ${SM_MODEL_DIR})"