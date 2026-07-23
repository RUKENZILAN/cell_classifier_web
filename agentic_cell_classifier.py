#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 Agentic Cell Culture Classifier  —  TEK DOSYA (Single-File) Sürüm
============================================================================
Hücre kültürü görüntülerini x5/x20 (veya istediğiniz sayıda) büyütme
seviyesinde kategorize eden ve hangi büyütmenin daha iyi sonuç verdiğine
otonom karar veren çok-ajanlı sistem. Tüm mimari (Central Memory, Intaker,
Analyzer, Decision Maker, Central Agent) bu tek dosyada.

KURULUM
-------
    pip install numpy scikit-learn pillow torch torchvision

    (torch/torchvision ağırsa ve sadece mekanizmayı denemek istiyorsanız
     kurmadan --dummy ile çalıştırabilirsiniz.)

KULLANIM — En kolay yol (interaktif):
    python agentic_cell_classifier.py
    -> veri klasörünü soracak, sizden Enter'a basmanızı isteyecek.

KULLANIM — Komut satırından:
    python agentic_cell_classifier.py --data-dir ./data --magnifications x5 x20

VERİ KLASÖR YAPISI (zorunlu):
    data/
      x5/
        Saglikli_Hucre/*.jpg
        Hastalikli_Hucre/*.jpg
      x20/
        Saglikli_Hucre/*.jpg
        Hastalikli_Hucre/*.jpg

ÇIKTI: ./outputs/final_report.md, ./outputs/run_memory.json
============================================================================
"""

import os
import sys
import json
import math
import hashlib
import argparse
import threading
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

import numpy as np


# ============================================================================
# 1) CONFIG
# ============================================================================

@dataclass
class PipelineConfig:
    data_dir: str
    magnifications: List[str] = field(default_factory=lambda: ["x5", "x20"])
    image_size: Tuple[int, int] = (224, 224)
    valid_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    backbone: str = "resnet50"          # "resnet50" | "vgg16" | "dummy"
    knn_neighbors: int = 5
    knn_metric: str = "cosine"

    test_size: float = 0.2
    random_state: int = 42

    cache_dir: str = "./.cache"
    output_dir: str = "./outputs"

    use_dummy_extractor: bool = False   # torch olmadan test için

    # Analiz eklentileri
    cv_folds: int = 0                          # 0 = kapalı; >=2 ise k-fold cross-validation çalışır
    max_misclassified_examples: int = 8        # yanlış sınıflandırılan görüntü galerisinde gösterilecek maksimum örnek
    thumbnail_size: Tuple[int, int] = (72, 72)  # galeri küçük resim boyutu

    def __post_init__(self):
        if not (0.0 < self.test_size < 1.0):
            raise ValueError("test_size 0 ile 1 arasında olmalı.")
        if len(self.magnifications) < 1:
            raise ValueError("En az bir büyütme (magnification) seviyesi belirtilmeli.")


# ============================================================================
# 2) CENTRAL MEMORY
# ============================================================================

class CentralMemory:
    """Tüm ajanların okuyup yazdığı ortak, thread-safe 'yazı tahtası'."""

    def __init__(self):
        self._lock = threading.Lock()
        self.storage: Dict[str, Any] = {
            "raw_data_path": None,
            "processed_data": {},
            "features": {},
            "metrics": {},
            "errors": [],
            "logs": [],
            "status": "initialized",
        }

    def write(self, key, value):
        with self._lock:
            self.storage[key] = value
        self.log(f"Memory Updated: {key}")

    def write_nested(self, top_key, sub_key, value):
        with self._lock:
            self.storage.setdefault(top_key, {})[sub_key] = value
        self.log(f"Memory Updated: {top_key}.{sub_key}")

    def read(self, key, default=None):
        with self._lock:
            return self.storage.get(key, default)

    def read_nested(self, top_key, sub_key, default=None):
        with self._lock:
            return self.storage.get(top_key, {}).get(sub_key, default)

    def log(self, message):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        with self._lock:
            self.storage["logs"].append(line)
        print(f"[MEMORY LOG] {line}")

    def report_error(self, agent, message):
        entry = {"agent": agent, "message": message, "timestamp": time.time()}
        with self._lock:
            self.storage["errors"].append(entry)
        self.log(f"ERROR in {agent}: {message}")

    def get_logs(self):
        with self._lock:
            return list(self.storage["logs"])

    def to_json(self, path):
        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [sanitize(v) for v in obj]
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            shape = getattr(obj, "shape", None)
            if shape is not None:
                return f"<array shape={tuple(shape)}>"
            return str(obj)

        with self._lock:
            snapshot = sanitize(self.storage)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)


# ============================================================================
# 3) FEATURE EXTRACTORS
# ============================================================================

class DummyFeatureExtractor:
    """Torch gerektirmeyen, hızlı, deterministik öznitelik çıkarıcı (test/geliştirme)."""
    name = "dummy"

    def extract(self, images: List[np.ndarray]) -> np.ndarray:
        feats = []
        for img in images:
            arr = np.asarray(img, dtype=np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            channel_means = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
            channel_stds = arr.reshape(-1, arr.shape[-1]).std(axis=0)
            hist = np.histogram(arr, bins=16, range=(0, 1))[0].astype(np.float32)
            hist = hist / (hist.sum() + 1e-8)
            feats.append(np.concatenate([channel_means, channel_stds, hist]))
        return np.stack(feats, axis=0)


class TorchBackboneFeatureExtractor:
    """Önceden eğitilmiş, dondurulmuş ResNet-50/VGG16 ile global öznitelik vektörleri çıkarır."""

    def __init__(self, backbone="resnet50", device=None, batch_size=32):
        import torch
        import torchvision.models as models
        import torch.nn as nn

        self.torch = torch
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name = backbone

        if backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            net = models.resnet50(weights=weights)
            net.fc = nn.Identity()
            self._preprocess = weights.transforms()
        elif backbone == "vgg16":
            weights = models.VGG16_Weights.IMAGENET1K_V1
            net = models.vgg16(weights=weights)
            net.classifier = nn.Sequential(*list(net.classifier.children())[:-3])
            self._preprocess = weights.transforms()
        else:
            raise ValueError(f"Bilinmeyen backbone: {backbone}")

        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        self.net = net.to(self.device)

    def extract(self, images: List[np.ndarray]) -> np.ndarray:
        torch = self.torch
        feats = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i + self.batch_size]
                tensors = []
                for img in batch:
                    arr = np.asarray(img, dtype=np.float32)
                    if arr.max() > 1.0:
                        arr = arr / 255.0
                    t = torch.from_numpy(arr).permute(2, 0, 1)
                    t = self._preprocess(t)
                    tensors.append(t)
                batch_tensor = torch.stack(tensors).to(self.device)
                out = self.net(batch_tensor)
                out = out.reshape(out.shape[0], -1)
                feats.append(out.cpu().numpy())
        return np.concatenate(feats, axis=0)


def build_feature_extractor(backbone: str, use_dummy: bool = False):
    if use_dummy or backbone == "dummy":
        return DummyFeatureExtractor()
    return TorchBackboneFeatureExtractor(backbone=backbone)


# ============================================================================
# 4) AGENTS
# ============================================================================

class BaseAgent:
    name = "BaseAgent"

    def __init__(self, memory: CentralMemory, config: PipelineConfig):
        self.memory = memory
        self.config = config

    def _fail(self, message: str):
        self.memory.report_error(self.name, message)
        raise RuntimeError(f"[{self.name}] {message}")


class IntakerAgent(BaseAgent):
    """Veri hazırlık: görüntüleri okur, boyutlandırır, normalize eder, train/test'e böler."""
    name = "IntakerAgent"

    def execute(self, magnification: str):
        from PIL import Image, UnidentifiedImageError
        from sklearn.model_selection import train_test_split

        mag_dir = os.path.join(self.config.data_dir, magnification)
        if not os.path.isdir(mag_dir):
            self._fail(f"Klasör bulunamadı: {mag_dir}")

        class_names = sorted(
            d for d in os.listdir(mag_dir) if os.path.isdir(os.path.join(mag_dir, d))
        )
        if len(class_names) < 2:
            self._fail(
                f"{magnification} altında en az 2 sınıf klasörü bekleniyor, "
                f"{len(class_names)} bulundu: {class_names}"
            )

        images, labels, paths, corrupted = [], [], [], 0
        for label_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(mag_dir, class_name)
            files = [f for f in os.listdir(class_dir) if f.lower().endswith(self.config.valid_extensions)]
            for fname in files:
                fpath = os.path.join(class_dir, fname)
                try:
                    with Image.open(fpath) as im:
                        im = im.convert("RGB").resize(self.config.image_size)
                        arr = np.asarray(im, dtype=np.float32) / 255.0
                    images.append(arr)
                    labels.append(label_idx)
                    paths.append(fpath)
                except (UnidentifiedImageError, OSError) as e:
                    corrupted += 1
                    self.memory.log(f"[{self.name}] Bozuk görüntü atlandı: {fpath} ({e})")

        if corrupted:
            self.memory.log(f"[{self.name}] {magnification}: {corrupted} bozuk görüntü atlandı.")
        if len(images) < 4:
            self._fail(f"{magnification} için yeterli sayıda geçerli görüntü yok (bulunan: {len(images)}).")

        X = np.stack(images, axis=0)
        y = np.array(labels, dtype=np.int64)
        paths = np.array(paths, dtype=object)
        stratify = y if np.bincount(y).min() >= 2 else None

        X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
            X, y, paths, test_size=self.config.test_size, random_state=self.config.random_state, stratify=stratify
        )

        self.memory.write_nested("processed_data", magnification, {
            "X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test,
            "paths_test": paths_test,
            "class_names": class_names, "n_total": len(images), "n_corrupted": corrupted,
        })
        self.memory.log(
            f"{self.name}: {magnification} tamamlandı — {len(X_train)} eğitim / "
            f"{len(X_test)} test örneği, sınıflar={class_names}."
        )


class AnalyzerAgent(BaseAgent):
    """Öznitelik çıkarımı (CNN backbone) + KNN sınıflandırma, sonuç cache'leme."""
    name = "AnalyzerAgent"

    def __init__(self, memory, config):
        super().__init__(memory, config)
        self._extractor = None

    def _get_extractor(self):
        if self._extractor is None:
            self._extractor = build_feature_extractor(self.config.backbone, self.config.use_dummy_extractor)
        return self._extractor

    def _cache_path(self, magnification, split, n):
        os.makedirs(self.config.cache_dir, exist_ok=True)
        key = f"{magnification}_{split}_{self.config.backbone}_{n}_{self.config.image_size}"
        digest = hashlib.md5(key.encode()).hexdigest()[:12]
        return os.path.join(self.config.cache_dir, f"feat_{magnification}_{split}_{digest}.npz")

    def _extract_with_cache(self, images, magnification, split):
        cache_file = self._cache_path(magnification, split, len(images))
        if os.path.exists(cache_file):
            self.memory.log(f"{self.name}: {magnification}/{split} özellikler cache'ten yüklendi.")
            return np.load(cache_file)["features"]
        extractor = self._get_extractor()
        feats = extractor.extract(list(images))
        np.savez_compressed(cache_file, features=feats)
        return feats

    def _thumbnail_b64(self, arr: np.ndarray) -> str:
        """Bir [0,1] float32 görüntü dizisinden küçük, base64 kodlu PNG üretir (galeri için)."""
        import io
        import base64
        from PIL import Image

        uint8_arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        im = Image.fromarray(uint8_arr).resize(self.config.thumbnail_size)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def execute(self, magnification: str):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.metrics import (
            accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support,
        )

        data = self.memory.read_nested("processed_data", magnification)
        if data is None:
            self._fail(f"{magnification} için işlenmiş veri bulunamadı (Intaker önce çalışmalı).")

        self.memory.log(f"{self.name}: {magnification} için öznitelik çıkarımı ({self.config.backbone}) ve KNN başlıyor...")

        t0 = time.time()
        try:
            X_train_feat = self._extract_with_cache(data["X_train"], magnification, "train")
            X_test_feat = self._extract_with_cache(data["X_test"], magnification, "test")
        except ImportError as e:
            self._fail(
                f"torch/torchvision bulunamadı ({e}). 'pip install torch torchvision' çalıştırın, "
                f"ya da --dummy ile test edin."
            )
        t1 = time.time()

        knn = KNeighborsClassifier(n_neighbors=self.config.knn_neighbors, metric=self.config.knn_metric)
        knn.fit(X_train_feat, data["y_train"])
        t2 = time.time()
        y_pred = knn.predict(X_test_feat)
        t3 = time.time()

        class_names = data["class_names"]
        y_test = data["y_test"]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        ci_low, ci_high = _wilson_interval(acc, len(y_test))

        # Sınıf bazlı precision / recall / F1 / support
        precisions, recalls, f1s, supports = precision_recall_fscore_support(
            y_test, y_pred, labels=list(range(len(class_names))), zero_division=0
        )
        per_class = [
            {
                "class_name": class_names[i],
                "precision": float(precisions[i]),
                "recall": float(recalls[i]),
                "f1_score": float(f1s[i]),
                "support": int(supports[i]),
            }
            for i in range(len(class_names))
        ]

        # Yanlış sınıflandırılan örnekler (küçük görsel galeri)
        misclassified = []
        wrong_idx = np.where(y_pred != y_test)[0]
        for idx in wrong_idx[: self.config.max_misclassified_examples]:
            try:
                thumb = self._thumbnail_b64(data["X_test"][idx])
            except Exception as e:
                self.memory.log(f"[{self.name}] Küçük resim oluşturulamadı: {e}")
                thumb = None
            path = data["paths_test"][idx] if "paths_test" in data else None
            misclassified.append({
                "true_class": class_names[int(y_test[idx])],
                "pred_class": class_names[int(y_pred[idx])],
                "filename": os.path.basename(path) if path else None,
                "thumbnail": thumb,
            })
        n_misclassified_total = int(len(wrong_idx))

        # k-fold cross-validation (opsiyonel)
        cv_result = None
        if self.config.cv_folds and self.config.cv_folds >= 2:
            X_all = np.concatenate([X_train_feat, X_test_feat], axis=0)
            y_all = np.concatenate([data["y_train"], y_test], axis=0)
            min_class_count = int(np.bincount(y_all).min())
            n_splits = min(self.config.cv_folds, min_class_count)
            if n_splits >= 2:
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_state)
                cv_knn = KNeighborsClassifier(n_neighbors=self.config.knn_neighbors, metric=self.config.knn_metric)
                scores = cross_val_score(cv_knn, X_all, y_all, cv=skf, scoring="accuracy")
                cv_result = {
                    "folds": n_splits,
                    "mean_accuracy": float(scores.mean()),
                    "std_accuracy": float(scores.std()),
                    "scores": [float(s) for s in scores],
                }
                self.memory.log(
                    f"{self.name}: {magnification} {n_splits}-fold CV accuracy = "
                    f"{scores.mean():.4f} ± {scores.std():.4f}"
                )
            else:
                self.memory.log(
                    f"[{self.name}] {magnification}: cross-validation atlandı "
                    f"(en küçük sınıf {min_class_count} örnek, en az 2 fold gerekli)."
                )
        t4 = time.time()

        metrics = {
            "accuracy": float(acc), "f1_score": float(f1), "confusion_matrix": cm,
            "ci_low": float(ci_low), "ci_high": float(ci_high),
            "class_names": class_names, "n_test_samples": len(y_test),
            "n_train_samples": len(data["y_train"]),
            "backbone": "dummy" if self.config.use_dummy_extractor else self.config.backbone,
            "knn_neighbors": self.config.knn_neighbors,
            "per_class": per_class,
            "misclassified": misclassified,
            "n_misclassified_total": n_misclassified_total,
            "cross_validation": cv_result,
            "timings": {
                "feature_extraction_sec": round(t1 - t0, 3),
                "train_sec": round(t2 - t1, 3),
                "inference_sec": round(t3 - t2, 3),
                "analysis_total_sec": round(t4 - t0, 3),
            },
        }

        self.memory.write_nested("features", magnification, {
            "train_shape": tuple(X_train_feat.shape), "test_shape": tuple(X_test_feat.shape),
        })
        self.memory.write_nested("metrics", magnification, metrics)
        self.memory.log(f"{self.name}: {magnification} tamamlandı — accuracy={acc:.4f}, f1={f1:.4f}")


def _wilson_interval(p: float, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) / n) + (z ** 2 / (4 * n ** 2)))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


class DecisionMakerAgent(BaseAgent):
    """Büyütme seviyelerini kıyaslar, gerekçeli markdown rapor üretir."""
    name = "DecisionMakerAgent"

    def execute(self) -> str:
        all_metrics = self.memory.read("metrics", {})
        if not all_metrics:
            self._fail("Karşılaştırılacak metrik bulunamadı.")

        ranked = sorted(all_metrics.items(), key=lambda kv: (kv[1]["accuracy"] + kv[1]["f1_score"]) / 2, reverse=True)
        best_mag, best_metrics = ranked[0]
        overlap = self._intervals_overlap(all_metrics)

        lines = ["# Final Cell Culture Categorization Report", "",
                 f"_Oluşturulma: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_", "",
                 "## Büyütme Seviyesi Karşılaştırması", "",
                 "| Büyütme | Accuracy | %95 Güven Aralığı | F1 (weighted) | Test N | CV (k-fold) |",
                 "|---|---|---|---|---|---|"]
        for mag, m in all_metrics.items():
            cv = m.get("cross_validation")
            cv_str = f"{cv['mean_accuracy']:.3f} ± {cv['std_accuracy']:.3f} ({cv['folds']}-fold)" if cv else "—"
            lines.append(
                f"| {mag} | {m['accuracy']:.4f} | [{m['ci_low']:.3f}, {m['ci_high']:.3f}] | "
                f"{m['f1_score']:.4f} | {m['n_test_samples']} | {cv_str} |"
            )
        lines.append("")

        lines.append("## Sınıf Bazlı Metrikler (Precision / Recall / F1 / Support)")
        lines.append("")
        for mag, m in all_metrics.items():
            lines.append(f"**{mag}**")
            lines.append("")
            lines.append("| Sınıf | Precision | Recall | F1 | Support |")
            lines.append("|---|---|---|---|---|")
            for pc in m.get("per_class", []):
                lines.append(
                    f"| {pc['class_name']} | {pc['precision']:.3f} | {pc['recall']:.3f} | "
                    f"{pc['f1_score']:.3f} | {pc['support']} |"
                )
            lines.append("")

        lines.append("## Confusion Matrix Detayları")
        lines.append("")
        for mag, m in all_metrics.items():
            lines.append(f"**{mag}** (sınıflar: {', '.join(m['class_names'])})")
            lines.append("")
            for row in m["confusion_matrix"]:
                lines.append(f"- {row}")
            lines.append("")

        lines.append("## Yanlış Sınıflandırılan Örnekler")
        lines.append("")
        for mag, m in all_metrics.items():
            n_wrong = m.get("n_misclassified_total", 0)
            shown = len(m.get("misclassified", []))
            lines.append(
                f"- **{mag}**: {n_wrong} / {m['n_test_samples']} test örneği yanlış sınıflandırıldı"
                + (f" (ilk {shown} tanesi için küçük resim galerisi UI'da mevcut)." if shown else ".")
            )
        lines.append("")

        lines.append("## Süre (Performans)")
        lines.append("")
        lines.append("| Büyütme | Öznitelik Çıkarımı | Eğitim | Çıkarım (inference) | Toplam |")
        lines.append("|---|---|---|---|---|")
        for mag, m in all_metrics.items():
            t = m.get("timings", {})
            lines.append(
                f"| {mag} | {t.get('feature_extraction_sec', '—')}s | {t.get('train_sec', '—')}s | "
                f"{t.get('inference_sec', '—')}s | {t.get('analysis_total_sec', '—')}s |"
            )
        lines.append("")

        lines.append("## Sonuç (Conclusion)")
        lines.append("")
        significance_note = (
            "Güven aralıkları çakışmıyor, fark istatistiksel olarak anlamlı görünüyor."
            if not overlap else
            "Güven aralıkları kısmen çakışıyor; daha büyük bir test seti ile doğrulama önerilir."
        )
        lines.append(
            f"**{best_mag}** büyütme seviyesi, {best_metrics['accuracy']:.1%} accuracy ve "
            f"{best_metrics['f1_score']:.1%} F1-skoru ile en iyi sonucu verdi. {significance_note}"
        )
        lines.append("")
        lines.append(
            "Yorum: Daha yüksek büyütme oranı genellikle hücresel sınır ve morfolojik detayları daha "
            "belirgin hale getirir, bu da CNN backbone'un ayırt edici öznitelik vektörleri çıkarmasını "
            "ve dolayısıyla KNN'in sınıf sınırlarını daha net ayırmasını kolaylaştırır. Düşük büyütmede "
            "ise daha geniş bir görüş alanı (doku bağlamı) elde edilir ama hücre detayları kaybolabilir; "
            "hangisinin baskın geldiği veri setine ve hedef sınıflara bağlıdır."
        )

        report = "\n".join(lines)
        os.makedirs(self.config.output_dir, exist_ok=True)
        report_path = os.path.join(self.config.output_dir, "final_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        self.memory.write("status", "completed")
        self.memory.write("report_path", report_path)
        self.memory.write("best_magnification", best_mag)
        self.memory.write("significance", {
            "significant": not overlap,
            "best_magnification": best_mag,
            "note": significance_note,
        })
        self.memory.log(f"{self.name}: Rapor oluşturuldu -> {report_path}")

        print(report)
        return report

    @staticmethod
    def _intervals_overlap(all_metrics) -> bool:
        intervals = sorted((m["ci_low"], m["ci_high"]) for m in all_metrics.values())
        for i in range(len(intervals) - 1):
            if intervals[i][1] >= intervals[i + 1][0]:
                return True
        return False


class CentralAgent(BaseAgent):
    """Pipeline'ı orkestre eder: Intaker -> Analyzer -> Decision Maker. Hatada durur."""
    name = "CentralAgent"

    def __init__(self, memory, config):
        super().__init__(memory, config)
        self.intaker = IntakerAgent(memory, config)
        self.analyzer = AnalyzerAgent(memory, config)
        self.decision_maker = DecisionMakerAgent(memory, config)

    def run_pipeline(self) -> str:
        self.memory.write("status", "running")
        self.memory.write("raw_data_path", self.config.data_dir)
        self.memory.log(
            f"{self.name}: Pipeline başlıyor — magnifications={self.config.magnifications}, backbone={self.config.backbone}"
        )
        try:
            for mag in self.config.magnifications:
                self.intaker.execute(mag)
            for mag in self.config.magnifications:
                self.analyzer.execute(mag)
            report = self.decision_maker.execute()
        except Exception as e:
            self.memory.write("status", "failed")
            self.memory.report_error(self.name, str(e))
            self.memory.log(f"{self.name}: Pipeline HATA nedeniyle durduruldu: {e}")
            self._write_failure_report(e)
            raise

        os.makedirs(self.config.output_dir, exist_ok=True)
        self.memory.to_json(os.path.join(self.config.output_dir, "run_memory.json"))
        self.memory.log(f"{self.name}: Pipeline başarıyla tamamlandı.")
        return report

    def _write_failure_report(self, error):
        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, "failure_report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Pipeline Failed\n\n")
            f.write(f"**Hata:** {error}\n\n")
            f.write("## Loglar\n\n")
            for line in self.memory.get_logs():
                f.write(f"- {line}\n")
        self.memory.to_json(os.path.join(self.config.output_dir, "run_memory.json"))


# ============================================================================
# 5) KOLAY ARAYÜZ (CLI + interaktif mod)
# ============================================================================

def _interactive_setup(args) -> PipelineConfig:
    """Argüman verilmeden çalıştırıldığında kullanıcıya soru sorarak konfigürasyonu kurar.
    Komut satırından verilmiş diğer bayraklar (--output-dir, --cache-dir, vb.) korunur."""
    print("=" * 70)
    print(" Agentic Cell Culture Classifier — İnteraktif Kurulum")
    print("=" * 70)

    data_dir = input("1) Veri klasörü yolu (örn. ./data) [Enter: ./data]: ").strip() or "./data"

    mags_raw = input("2) Büyütme seviyeleri, boşlukla ayırın [Enter: x5 x20]: ").strip()
    magnifications = mags_raw.split() if mags_raw else ["x5", "x20"]

    dummy_raw = input("3) torch/torchvision kurulu mu? (e/h) [Enter: e]: ").strip().lower()
    use_dummy = dummy_raw == "h" or args.dummy
    if use_dummy:
        print("   -> Not: dummy (test) extractor kullanılacak (gerçek model değil).")

    return PipelineConfig(
        data_dir=data_dir,
        magnifications=magnifications,
        backbone=args.backbone,
        knn_neighbors=args.knn_neighbors,
        test_size=args.test_size,
        random_state=args.random_state,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        use_dummy_extractor=use_dummy,
        cv_folds=args.cv_folds,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Hücre Kültürü Büyütme Kıyaslama Pipeline'ı (tek dosya)")
    p.add_argument("--data-dir", default=None, help="Kök veri klasörü (data/x5/..., data/x20/...)")
    p.add_argument("--magnifications", nargs="+", default=None)
    p.add_argument("--backbone", default="resnet50", choices=["resnet50", "vgg16", "dummy"])
    p.add_argument("--knn-neighbors", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--cache-dir", default="./.cache")
    p.add_argument("--output-dir", default="./outputs")
    p.add_argument("--dummy", action="store_true", help="torch olmadan pipeline mantığını test etmek için")
    p.add_argument("--cv-folds", type=int, default=0, help="k-fold cross-validation (0=kapalı, >=2 açık)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.data_dir is None:
        # Hiç argüman verilmemişse en kolay yol: interaktif soru-cevap
        config = _interactive_setup(args)
    else:
        config = PipelineConfig(
            data_dir=args.data_dir,
            magnifications=args.magnifications or ["x5", "x20"],
            backbone=args.backbone,
            knn_neighbors=args.knn_neighbors,
            test_size=args.test_size,
            random_state=args.random_state,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            use_dummy_extractor=args.dummy,
            cv_folds=args.cv_folds,
        )

    memory = CentralMemory()
    orchestrator = CentralAgent(memory, config)
    try:
        orchestrator.run_pipeline()
        print(f"\n✅ Tamamlandı. Rapor: {os.path.join(config.output_dir, 'final_report.md')}")
    except Exception as e:
        print(f"\n❌ Pipeline durdu: {e}")
        print(f"   Detaylar: {os.path.join(config.output_dir, 'failure_report.md')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
