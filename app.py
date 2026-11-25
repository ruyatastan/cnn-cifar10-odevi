import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from PIL import Image, ImageOps
from torchvision import models, transforms

classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNet18Cifar(nn.Module):
    def __init__(self, num_classes=10, img_size=128):
        super().__init__()
        self.img_size = img_size
        self.net = models.resnet18(weights=None)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x):
        x = F.interpolate(x, size=self.img_size, mode="bilinear", align_corners=False)
        return self.net(x)


model = ResNet18Cifar().to(device)
model.load_state_dict(torch.load("resnet18_best.pth", map_location=device))
model.eval()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


def predict(image):
    if image is None:
        return None, ""
    img = ImageOps.exif_transpose(image).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((32, 32), Image.LANCZOS)

    x = preprocess(img).unsqueeze(0).to(device)
    start = time.time()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu()
    ms = (time.time() - start) * 1000
    return {classes[i]: float(probs[i]) for i in range(10)}, f"{ms:.1f} ms"


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Fotoğraf yükle"),
    outputs=[gr.Label(num_top_classes=3, label="Tahmin"),
             gr.Textbox(label="Tahmin süresi")],
    title="CIFAR-10 Görüntü Sınıflandırma (ResNet18)",
    description="Bir fotoğraf yükle, model sınıfını tahmin etsin. Sınıflar: " + ", ".join(classes),
)

if __name__ == "__main__":
    demo.launch()
