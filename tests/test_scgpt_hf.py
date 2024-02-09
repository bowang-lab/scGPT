import sys

sys.path.insert(0, "../")

# from scgpt.huggingface_model import scGPT_config, scGPT_ForPretraining
from scgpt.model.huggingface_model import scGPT_config, scGPT_ForPretraining


config = scGPT_config()
model = scGPT_ForPretraining(config)

print(model)
