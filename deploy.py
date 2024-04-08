"""
this version is modified from https://pytorch.org/mobile/android/
"""
import torch
import torchvision

from model import get_model
from utils import load_best_model_state_dict
from torch.utils.mobile_optimizer import optimize_for_mobile

model = get_model()
state_dict = load_best_model_state_dict("model")
model.load_state_dict(state_dict["model"])
model.eval()

example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module_optimized._save_for_lite_interpreter("model/model.ptl")
