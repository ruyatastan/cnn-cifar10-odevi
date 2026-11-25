# CIFAR-10 Görüntü Sınıflandırma (CNN ve ResNet18)

Computer Vision ödevi: CIFAR-10 veri seti ile görüntü sınıflandırma, Gradio ile uygulama.

## Özet
- **Görev:** Görüntü sınıflandırma (10 sınıf)
- **Veri seti:** CIFAR-10 (45.000 eğitim / 5.000 doğrulama / 10.000 test)
- **Modeller:** SimpleCNN (sıfırdan) ve ResNet18 (transfer learning)
- **Optimizer / Loss:** Adam / CrossEntropyLoss
- **Ortam:** Google Colab, Tesla T4 GPU

| Model | Test Accuracy | Weighted F1 | Eğitim süresi | Tahmin süresi |
|---|---|---|---|---|
| SimpleCNN | %77,03 | 0,769 | 4,2 dk | 0,61 ms |
| ResNet18 | %93,88 | 0,939 | 4,3 dk | 2,11 ms |

## Dosyalar
- `notebook/` : Eğitim ve değerlendirme notebook'u
- `app.py` : Gradio uygulaması (ResNet18 kullanır)
- `requirements.txt` : Gerekli kütüphaneler

## Çalıştırma
1. Notebook'u çalıştırıp `resnet18_best.pth` dosyasını üret (ya da kendi model dosyanı `app.py` ile aynı klasöre koy).
2. `pip install -r requirements.txt`
3. `python app.py`

Not: Model dosyası boyutu nedeniyle repoya eklenmemiştir.
