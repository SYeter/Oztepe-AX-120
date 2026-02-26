from __future__ import annotations

import sys
from dataclasses import dataclass

import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

WINDOW_TITLE = "Enjeksiyon Kamera Kontrol"


@dataclass
class AppState:
    yolluk_roi: tuple[int, int, int, int] | None = None
    urun_roi: tuple[int, int, int, int] | None = None
    selected_bgr: tuple[int, int, int] | None = None
    expected_count: int = 1
    threshold_percent: int = 90


class VideoLabel(QLabel):
    """Kamera görüntüsü üstünde ROI ve renk seçimi için tıklama/drag desteği."""

    def __init__(self, parent: "MainWindow") -> None:
        super().__init__(parent)
        self.main_window = parent
        self.start_point: tuple[int, int] | None = None
        self.current_point: tuple[int, int] | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(960, 540)
        self.setStyleSheet(
            "QLabel {"
            " background: #11151d;"
            " border: 2px solid #253042;"
            " border-radius: 12px;"
            "}"
        )

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.LeftButton:
            return

        x, y = event.x(), event.y()
        if self.main_window.selection_mode in ("yolluk", "urun"):
            self.start_point = (x, y)
            self.current_point = (x, y)
            return

        if self.main_window.selection_mode == "color":
            self.main_window.pick_color_at(x, y)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self.start_point is not None:
            self.current_point = (event.x(), event.y())

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self.start_point is None:
            return

        end = (event.x(), event.y())
        roi = normalize_roi(self.start_point, end)
        self.start_point = None
        self.current_point = None

        if roi[2] < 10 or roi[3] < 10:
            self.main_window.update_status("⚠️ Geçersiz ROI: daha büyük bir alan seçin.")
            self.main_window.selection_mode = None
            return

        self.main_window.assign_roi(roi)


class MainWindow(QWidget):
    def __init__(self, cam_index: int) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)

        self.state = AppState()
        self.selection_mode: str | None = None
        self.current_frame: np.ndarray | None = None

        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError("Kamera açılamadı. Kamera indeksi veya erişim yetkisini kontrol edin.")

        self.video_label = VideoLabel(self)

        self.status_label = QLabel("Hazır. ROI veya renk seçimi için aşağıdaki butonları kullanın.")
        self.status_label.setObjectName("status")

        self.expected_input = QLineEdit("1")
        self.threshold_input = QLineEdit("90")

        self.metric_selected_color = QLabel("Seçili Renk (BGR): -")
        self.metric_count = QLabel("Anlık Ürün: 0")
        self.metric_signal = QLabel("Çıkış Sinyali: 0")
        self.metric_yolluk = QLabel("Yolluk: YOK")

        self.init_ui()
        self.start_timer()

    def init_ui(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background-color: #0f141d;
                color: #e9eef8;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #2d3a4f;
                border-radius: 12px;
                margin-top: 10px;
                padding-top: 12px;
                background-color: #171f2d;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #9fb3d8;
                font-weight: bold;
            }
            QPushButton {
                background-color: #28364c;
                border: 1px solid #3d5374;
                border-radius: 10px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #324666; }
            QPushButton:pressed { background-color: #223149; }
            QLineEdit {
                background-color: #111925;
                border: 1px solid #3a4b67;
                border-radius: 8px;
                padding: 8px;
                color: #f4f7ff;
            }
            QLabel#status {
                background-color: #182437;
                border: 1px solid #2c466e;
                border-radius: 8px;
                padding: 10px;
                color: #cfe1ff;
            }
            """
        )

        roi_group = QGroupBox("Alan ve Renk Seçimi")
        roi_layout = QHBoxLayout()

        yolluk_btn = QPushButton("Yolluk Alanı Seç")
        urun_btn = QPushButton("Ürün Alanı Seç")
        color_btn = QPushButton("Renk Seç")

        yolluk_btn.clicked.connect(lambda: self.activate_mode("yolluk"))
        urun_btn.clicked.connect(lambda: self.activate_mode("urun"))
        color_btn.clicked.connect(lambda: self.activate_mode("color"))

        roi_layout.addWidget(yolluk_btn)
        roi_layout.addWidget(urun_btn)
        roi_layout.addWidget(color_btn)
        roi_group.setLayout(roi_layout)

        settings_group = QGroupBox("Üretim Parametreleri")
        settings_layout = QGridLayout()
        settings_layout.addWidget(QLabel("Beklenen Ürün Adedi"), 0, 0)
        settings_layout.addWidget(self.expected_input, 0, 1)
        settings_layout.addWidget(QLabel("Verim Eşiği (%)"), 1, 0)
        settings_layout.addWidget(self.threshold_input, 1, 1)

        apply_btn = QPushButton("Değerleri Uygula")
        apply_btn.clicked.connect(self.apply_inputs)
        settings_layout.addWidget(apply_btn, 0, 2, 2, 1)
        settings_group.setLayout(settings_layout)

        metrics_group = QGroupBox("Canlı Sonuçlar")
        metrics_layout = QVBoxLayout()
        for metric in [self.metric_selected_color, self.metric_count, self.metric_signal, self.metric_yolluk]:
            card = QFrame()
            card.setStyleSheet(
                "QFrame {"
                " background-color: #111925;"
                " border: 1px solid #2f3f59;"
                " border-radius: 8px;"
                " padding: 6px;"
                "}"
            )
            row = QHBoxLayout(card)
            row.addWidget(metric)
            metrics_layout.addWidget(card)

        metrics_group.setLayout(metrics_layout)

        left_panel = QVBoxLayout()
        left_panel.addWidget(roi_group)
        left_panel.addWidget(settings_group)
        left_panel.addWidget(metrics_group)
        left_panel.addWidget(self.status_label)

        root = QHBoxLayout()
        root.addLayout(left_panel, 3)
        root.addWidget(self.video_label, 7)
        self.setLayout(root)
        self.resize(1500, 850)

    def start_timer(self) -> None:
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def apply_inputs(self) -> None:
        try:
            expected = int(self.expected_input.text())
            threshold = int(self.threshold_input.text())
        except ValueError:
            QMessageBox.warning(self, "Hatalı Giriş", "Lütfen sadece sayısal değer girin.")
            return

        self.state.expected_count = max(1, min(999, expected))
        self.state.threshold_percent = max(0, min(100, threshold))

        self.expected_input.setText(str(self.state.expected_count))
        self.threshold_input.setText(str(self.state.threshold_percent))
        self.update_status("✅ Parametreler güncellendi.")

    def activate_mode(self, mode: str) -> None:
        self.selection_mode = mode
        messages = {
            "yolluk": "Yolluk ROI modu aktif. Görüntü üzerinde sürükleyerek alan seçin.",
            "urun": "Ürün ROI modu aktif. Görüntü üzerinde sürükleyerek alan seçin.",
            "color": "Renk seçimi aktif. Görüntüde bir piksele tıklayın.",
        }
        self.update_status(messages.get(mode, "Mod değiştirildi."))

    def assign_roi(self, roi: tuple[int, int, int, int]) -> None:
        if self.selection_mode == "yolluk":
            self.state.yolluk_roi = roi
            self.update_status(f"✅ Yolluk alanı tanımlandı: {roi}")
        elif self.selection_mode == "urun":
            self.state.urun_roi = roi
            self.update_status(f"✅ Ürün alanı tanımlandı: {roi}")
        self.selection_mode = None

    def pick_color_at(self, x: int, y: int) -> None:
        if self.current_frame is None:
            return
        if not (0 <= y < self.current_frame.shape[0] and 0 <= x < self.current_frame.shape[1]):
            return

        b, g, r = self.current_frame[y, x]
        self.state.selected_bgr = (int(b), int(g), int(r))
        self.metric_selected_color.setText(f"Seçili Renk (BGR): {self.state.selected_bgr}")
        self.update_status(f"✅ Renk seçildi: {self.state.selected_bgr}")
        self.selection_mode = None

    def update_status(self, message: str) -> None:
        self.status_label.setText(message)

    def update_frame(self) -> None:
        ret, frame = self.cap.read()
        if not ret:
            self.update_status("❌ Kameradan görüntü alınamadı.")
            return

        self.current_frame = frame.copy()

        if self.video_label.start_point and self.video_label.current_point:
            cv2.rectangle(frame, self.video_label.start_point, self.video_label.current_point, (0, 165, 255), 2)

        urun_sayisi, yolluk_var, debug_frame = count_products_and_yolluk(frame, self.state)

        minimum_required = self.state.expected_count * (self.state.threshold_percent / 100.0)
        signal = 1 if urun_sayisi < minimum_required else 0

        self.metric_count.setText(f"Anlık Ürün: {urun_sayisi} | Beklenen: {self.state.expected_count}")
        self.metric_signal.setText(f"Çıkış Sinyali: {signal} | Eşik: %{self.state.threshold_percent}")
        self.metric_signal.setStyleSheet(f"color: {'#ff6f6f' if signal else '#66df8f'};")
        self.metric_yolluk.setText(f"Yolluk: {'VAR' if yolluk_var else 'YOK'}")

        rgb = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


def normalize_roi(p1: tuple[int, int], p2: tuple[int, int]) -> tuple[int, int, int, int]:
    x1, y1 = p1
    x2, y2 = p2
    x_min, x_max = sorted((x1, x2))
    y_min, y_max = sorted((y1, y2))
    return x_min, y_min, x_max - x_min, y_max - y_min


def draw_roi(frame: np.ndarray, roi: tuple[int, int, int, int], color: tuple[int, int, int], label: str) -> None:
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def build_mask_by_selected_color(frame: np.ndarray, selected_bgr: tuple[int, int, int], tol: int = 35) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_pixel = np.uint8([[selected_bgr]])
    selected_hsv = cv2.cvtColor(color_pixel, cv2.COLOR_BGR2HSV)[0][0]

    h, s, v = [int(c) for c in selected_hsv]

    lower = np.array([max(h - tol, 0), max(s - 70, 30), max(v - 70, 30)])
    upper = np.array([min(h + tol, 179), min(s + 70, 255), min(v + 70, 255)])

    return cv2.inRange(hsv, lower, upper)


def count_products_and_yolluk(frame: np.ndarray, state: AppState) -> tuple[int, bool, np.ndarray]:
    debug = frame.copy()
    if state.selected_bgr is None:
        if state.yolluk_roi:
            draw_roi(debug, state.yolluk_roi, (255, 0, 0), "Yolluk")
        if state.urun_roi:
            draw_roi(debug, state.urun_roi, (0, 255, 0), "Ürün")
        return 0, False, debug

    mask = build_mask_by_selected_color(frame, state.selected_bgr)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    yolluk_var = False
    if state.yolluk_roi:
        x, y, w, h = state.yolluk_roi
        yolluk_mask = mask[y:y + h, x:x + w]
        if yolluk_mask.size > 0:
            area_ratio = cv2.countNonZero(yolluk_mask) / yolluk_mask.size
            yolluk_var = area_ratio > 0.03
        draw_roi(debug, state.yolluk_roi, (255, 0, 0), f"Yolluk {'VAR' if yolluk_var else 'YOK'}")

    urun_sayisi = 0
    if state.urun_roi:
        x, y, w, h = state.urun_roi
        urun_mask = mask[y:y + h, x:x + w]

        contours, _ = cv2.findContours(urun_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 120:
                urun_sayisi += 1
                bx, by, bw, bh = cv2.boundingRect(cnt)
                cv2.rectangle(debug, (x + bx, y + by), (x + bx + bw, y + by + bh), (0, 255, 255), 2)

        draw_roi(debug, state.urun_roi, (0, 255, 0), f"Ürün Sayısı: {urun_sayisi}")

    return urun_sayisi, yolluk_var, debug


def main() -> None:
    cam_text = input("Kamera indeksi (varsayılan 0): ").strip() or "0"
    cam_index = int(cam_text)

    app = QApplication(sys.argv)
    window = MainWindow(cam_index)
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
