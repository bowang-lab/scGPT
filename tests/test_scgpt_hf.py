import sys

sys.path.insert(0, "/home/pangkuan/dev/scGPT-release/")

# from scgpt.huggingface_model import scGPT_config, scGPT_ForPretraining
# from scgpt.model.huggingface_model import scGPT_config, scGPT_ForPretraining

# print all submodules of scgpt
import scgpt

print(dir(scgpt.model))

config = scGPT_config()
model = scGPT_ForPretraining(config)
