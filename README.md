# Agentic Cell Culture Classifier

Hücre kültürü görüntülerini farklı büyütme seviyelerinde (x5, x20, …) sınıflandıran ve **hangi büyütme seviyesinin daha iyi sonuç verdiğine otonom karar veren** çok ajanlı (multi-agent) bir görüntü sınıflandırma sistemi. Hem komut satırından hem de bir Flask tabanlı web arayüzünden çalıştırılabilir.

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Mimari](#mimari)
- [Kurulum](#kurulum)
- [Veri Klasör Yapısı](#veri-klasör-yapısı)
- [Kullanım](#kullanım)
  - [Web arayüzü](#web-arayüzü)
  - [Komut satırı](#komut-satırı)
- [Yapılandırma Parametreleri](#yapılandırma-parametreleri)
- [Dummy vs. Gerçek Backbone](#dummy-vs-gerçek-backbone)
- [Çıktılar](#çıktılar)
- [Web API Uç Noktaları](#web-api-uç-noktaları)
- [Örnek Veri Üretimi](#örnek-veri-üretimi)
- [Sorun Giderme](#sorun-giderme)

## Genel Bakış

Sistem, her biri belirli bir sorumluluğu olan ajanlardan oluşur (Central Memory üzerinden haberleşirler):

1. Görüntüleri okur, boyutlandırır, normalize eder ve train/test setlerine böler.
2. Önceden eğitilmiş bir CNN (ResNet-50 / VGG-16) veya hızlı bir "dummy" çıkarıcı ile öznitelik (feature) çıkarır.
3. K-En Yakın Komşu (KNN) ile sınıflandırma yapar, metrikleri (accuracy, F1, confusion matrix, per-class sonuçlar) hesaplar.
4. İsteğe bağlı k-fold cross-validation ile güven aralıklarını daraltır.
5. Büyütme seviyeleri arasında karşılaştırma yaparak en iyi performans gösteren büyütmeye otonom karar verir ve bir final rapor (`final_report.md`) üretir.

## Mimari

```
CentralAgent (orkestratör)
 ├── IntakerAgent       → veri okuma, boyutlandırma, train/test bölme
 ├── AnalyzerAgent      → öznitelik çıkarma + KNN sınıflandırma + metrikler
 └── DecisionMakerAgent → büyütmeler arası karşılaştırma ve karar

CentralMemory  → tüm ajanların okuyup yazdığı thread-safe ortak durum (loglar, hatalar, metrikler, sonuçlar)
```

Tüm çekirdek mantık **tek dosyada** toplanmıştır: `agentic_cell_classifier.py`. `app.py` bu dosyayı import ederek üzerine bir Flask web arayüzü ve REST API'si ekler.

## Kurulum

```bash
# Temel bağımlılıklar
pip install flask numpy scikit-learn pillow

# Gerçek CNN backbone'ları (ResNet-50 / VGG-16) için — opsiyonel ama önerilir
pip install torch torchvision
```

> torch/torchvision kurmak istemiyorsanız, sistemi **dummy** öznitelik çıkarıcı ile de deneyebilirsiniz (bkz. [Dummy vs. Gerçek Backbone](#dummy-vs-gerçek-backbone)). Bu, mekanizmayı hızlıca test etmek için uygundur ama gerçek sınıflandırma performansını yansıtmaz.

## Veri Klasör Yapısı

Veri klasörünüz aşağıdaki gibi düzenlenmelidir — her büyütme seviyesi için bir alt klasör, her sınıf için de onun altında bir alt klasör:

```
data/
  x5/
    Saglikli_Hucre/*.jpg
    Hastalikli_Hucre/*.jpg
  x20/
    Saglikli_Hucre/*.jpg
    Hastalikli_Hucre/*.jpg
```

Desteklenen dosya uzantıları: `.png .jpg .jpeg .tif .tiff .bmp`

Kendi verinize hızlı bir alternatif olarak, sentetik/deneme verisi üretmek için `generate_sample_data.py` betiğini kullanabilirsiniz (bkz. [Örnek Veri Üretimi](#örnek-veri-üretimi)).

## Kullanım

### Web arayüzü

```bash
python app.py
# Tarayıcıda açın: http://127.0.0.1:5000
```

Arayüzden:

1. **1. Select Data** — `data/` klasörünüzü seçin (yerel dosya seçici ile) veya sunucudaki bir yolu doğrudan yazın.
2. **2. Configure** — büyütme seviyeleri, backbone, KNN k değeri, test oranı, çıktı klasörü ve cross-validation fold sayısını ayarlayın.
3. **▶ Run Pipeline** — çalıştırın; canlı loglar, metrikler, confusion matrix, sınıf bazlı sonuçlar, yanlış sınıflandırılan örnekler ve final rapor arayüzde anlık olarak güncellenir.
4. Sonuçları PNG/CSV/JSON/Markdown olarak dışa aktarabilir veya tüm görselleri tek bir ZIP olarak indirebilirsiniz.

### Komut satırı

**İnteraktif mod:**
```bash
python agentic_cell_classifier.py
```
Sizden veri klasörünü ve torch/torchvision'ın kurulu olup olmadığını soracaktır.

**Doğrudan parametrelerle:**
```bash
python agentic_cell_classifier.py --data-dir ./data --magnifications x5 x20 --backbone resnet50
```

Dummy (torch gerektirmeyen) modda çalıştırmak için:
```bash
python agentic_cell_classifier.py --data-dir ./data --dummy
```

## Yapılandırma Parametreleri

| Parametre | Açıklama | Varsayılan |
|---|---|---|
| `data_dir` | Veri klasörü yolu | — (zorunlu) |
| `magnifications` | Karşılaştırılacak büyütme seviyeleri | `["x5", "x20"]` |
| `backbone` | `resnet50` \| `vgg16` \| `dummy` | `resnet50` |
| `use_dummy_extractor` | torch olmadan hızlı test modu | `False` |
| `knn_neighbors` | KNN komşu sayısı (k) | `5` |
| `knn_metric` | KNN mesafe metriği | `cosine` |
| `test_size` | Test setinin oranı (0–1 arası) | `0.2` |
| `cv_folds` | K-fold cross-validation fold sayısı (0 = kapalı) | `0` |
| `output_dir` | Rapor ve çıktıların yazılacağı klasör | `./outputs` |
| `random_state` | Tekrarlanabilirlik için sabit tohum | `42` |
| `max_misclassified_examples` | Galeri başına gösterilecek maksimum yanlış örnek | `8` |

> **İstatistiksel not:** Küçük test setlerinde (sınıf başına ~20–30 örnek) güven aralıkları büyütmeler arasında örtüşebilir. Daha güvenilir bir karşılaştırma için `cv_folds` değerini 5 veya 10 yaparak k-fold cross-validation kullanmanız önerilir; bu, ek veri toplamadan aynı veri setiyle daha dar güven aralıkları elde etmenizi sağlar.

## Dummy vs. Gerçek Backbone

| | **Dummy** | **ResNet-50 / VGG-16** |
|---|---|---|
| Bağımlılık | Sadece numpy | torch + torchvision |
| Öznitelikler | Piksel kanal ortalaması/std'si + histogram | ImageNet üzerinde önceden eğitilmiş derin CNN öznitelikleri |
| Hız | Çok hızlı | Yavaş (CPU) / hızlı (GPU) |
| Kullanım amacı | Mekanizmayı test etmek, geliştirme | Gerçek sınıflandırma performansı |
| Doğruluk | Gerçek performansı yansıtmaz | Gerçek performansı yansıtır |

Gerçek bir backbone'a geçmek için:
1. `pip install torch torchvision`
2. Web arayüzünde **"Force dummy extractor"** kutusunun işaretini kaldırın ve **Backbone** olarak `ResNet-50` veya `VGG-16` seçin.
3. CLI'da `--backbone resnet50` (veya `vgg16`) kullanın ve `--dummy` bayrağını **eklemeyin**.

İlk çalıştırmada ImageNet ön-eğitimli ağırlıklar internetten indirilir (birkaç yüz MB); sonraki çalıştırmalarda yerel önbellekten (`~/.cache/torch`) okunur.

## Çıktılar

Belirtilen `output_dir` içine (varsayılan `./outputs`):

- `final_report.md` — okunabilir özet rapor (metrikler, karar, gerekçe)
- `run_memory.json` — çalışmanın tam durum kaydı (loglar, hatalar, ham metrikler)
- Confusion matrix, accuracy/F1 grafikleri ve yanlış sınıflandırılan örnek galerileri (web arayüzünden PNG/ZIP olarak indirilebilir)

## Web API Uç Noktaları

| Uç nokta | Metod | Açıklama |
|---|---|---|
| `/` | GET | Web arayüzü |
| `/api/upload` | POST | Yerel klasör yükleme |
| `/api/run` | POST | Pipeline'ı başlatır |
| `/api/logs` | GET | Canlı log akışı (SSE) |
| `/api/status` | GET | Mevcut çalışma durumu |
| `/api/cancel` | POST | Çalışan işi iptal eder |
| `/api/resolve-output` | POST | Çıktı klasörü yolunu çözümler |
| `/api/export/zip` | GET | Tüm görselleri ZIP olarak indirir |
| `/api/export/metrics.csv` | GET | Metrikleri CSV olarak indirir |
| `/api/export/results.json` | GET | Sonuçları JSON olarak indirir |
| `/api/export/report.md` | GET | Final raporu Markdown olarak indirir |
| `/api/reset` | POST | Arayüzü ve durumu sıfırlar |

## Örnek Veri Üretimi

Kendi veriniz yoksa, arayüzü/pipeline'ı denemek için küçük bir sentetik veri seti oluşturabilirsiniz:

```bash
python generate_sample_data.py
```

Bu, `./data/x5/` ve `./data/x20/` altında `class_a` / `class_b` klasörlerinde, sınıf başına 12 sentetik görüntü (renkli daireler) üretir.

## Sorun Giderme

| Sorun | Olası neden / çözüm |
|---|---|
| "Choose a folder" veya tema (dark/light) düğmesi tıklanmıyor | `index.html` içinde JS başlatma hatası olabilir; sayfayı yeniden yükleyin, tarayıcı konsolunda hata olup olmadığını kontrol edin. |
| `ModuleNotFoundError: torch` | `pip install torch torchvision` çalıştırın veya `use_dummy` / `--dummy` seçeneğini kullanın. |
| Güven aralıkları örtüşüyor | `cv_folds` ≥ 5 ile cross-validation kullanın veya test setini büyütün (bkz. Yapılandırma Parametreleri notu). |
| Yükleme sırasında büyük klasörlerde zaman aşımı | Sunucu tarafı yol (`data_dir`) kullanarak yükleme adımını atlayın. |

---

*Bu proje eğitim/araştırma amaçlı bir referans uygulamadır; klinik teşhis için kullanılmamalıdır.*
