from __future__ import print_function
from __future__ import division
import torch
import torchvision
from torchvision import datasets, models, transforms
import torchvision
import onnx
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
model = models.resnet50(pretrained=True)
model.eval()


# convert to ONNX --------------------------------------------------------------------------------------------------
ONNX_FILE_PATH = "resnet50.onnx"
input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, input, ONNX_FILE_PATH, input_names=["input"], output_names=["output"], export_params=True)

onnx_model = onnx.load(ONNX_FILE_PATH)
# check that the model converted fine
onnx.checker.check_model(onnx_model)

print("Model was successfully converted to ONNX format.")
print("It was saved to", ONNX_FILE_PATH)