import cv2
import numpy as np
from dataclasses import dataclass


WINDOW_NAME = "Enjeksiyon Kamera Kontrol"


@dataclass
class AppState:
    selecting_roi: str | None = None
    selecting_color: bool = False
    yolluk_roi: tuple[int, int, int, int] | None = None
    urun_roi: tuple[int, int, int, int] | None = None
    start_point: tuple[int, int] | None = None
    current_point: tuple[int, int] | None = None
    selected_bgr: tuple[int, int, int] | None = None
    expected_count: int = 1
    threshold_percent: int = 90


@dataclass
class UIButtons:
    yolluk_select: tuple[int, int, int, int]
    urun_select: tuple[int, int, int, int]
    color_select: tuple[int, int, int, int]


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


def point_in_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def draw_button(frame: np.ndarray, rect: tuple[int, int, int, int], text: str, is_active: bool = False) -> None:
    x, y, w, h = rect
    fill = (90, 170, 90) if is_active else (70, 70, 70)
    border = (180, 255, 180) if is_active else (170, 170, 170)
    cv2.rectangle(frame, (x, y), (x + w, y + h), fill, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), border, 2)
    cv2.putText(frame, text, (x + 10, y + int(h * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_ui_panel(frame: np.ndarray, state: AppState, buttons: UIButtons) -> None:
    panel_h = 130
    cv2.rectangle(frame, (0, 0), (frame.shape[1], panel_h), (35, 35, 35), -1)

    draw_button(frame, buttons.yolluk_select, "Yolluk Alani Sec", state.selecting_roi == "yolluk")
    draw_button(frame, buttons.urun_select, "Urun Alani Sec", state.selecting_roi == "urun")
    draw_button(frame, buttons.color_select, "Renk Sec", state.selecting_color)

    cv2.putText(frame, f"Beklenen Urun: {state.expected_count}", (15, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Verim Esigi: %{state.threshold_percent}", (260, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Trackbar ile degerleri ayarla", (15, 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def _noop(_: int) -> None:
    return


def on_mouse(event: int, x: int, y: int, flags: int, param: dict) -> None:
    state: AppState = param["state"]
    frame_ref: dict = param["frame_ref"]
    buttons: UIButtons = param["buttons"]

    if event == cv2.EVENT_LBUTTONDOWN:
        if point_in_rect(x, y, buttons.yolluk_select):
            state.selecting_roi = "yolluk"
            state.selecting_color = False
            print("[Bilgi] Yolluk ROI seçimi aktif. Sol tık + sürükle bırak.")
            return
        if point_in_rect(x, y, buttons.urun_select):
            state.selecting_roi = "urun"
            state.selecting_color = False
            print("[Bilgi] Ürün ROI seçimi aktif. Sol tık + sürükle bırak.")
            return
        if point_in_rect(x, y, buttons.color_select):
            state.selecting_color = True
            state.selecting_roi = None
            print("[Bilgi] Renk seçimi aktif. Görüntüden bir piksele tıklayın.")
            return

        if state.selecting_roi:
            state.start_point = (x, y)
            state.current_point = (x, y)
        elif state.selecting_color:
            frame = frame_ref.get("frame")
            if frame is not None and 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                b, g, r = frame[y, x]
                state.selected_bgr = int(b), int(g), int(r)
                state.selecting_color = False
                print(f"[Bilgi] Renk seçildi (BGR): {state.selected_bgr}")

    elif event == cv2.EVENT_MOUSEMOVE and state.start_point is not None:
        state.current_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP and state.start_point is not None:
        end_point = (x, y)
        roi = normalize_roi(state.start_point, end_point)
        if roi[2] > 10 and roi[3] > 10:
            if state.selecting_roi == "yolluk":
                state.yolluk_roi = roi
                print(f"[Bilgi] Yolluk alanı tanımlandı: {roi}")
            elif state.selecting_roi == "urun":
                state.urun_roi = roi
                print(f"[Bilgi] Ürün alanı tanımlandı: {roi}")
        else:
            print("[Uyarı] Geçersiz alan seçimi, daha büyük bir alan çizin.")

        state.start_point = None
        state.current_point = None
        state.selecting_roi = None


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
    print("=== Enjeksiyon Makinesi Görüntü Kontrol Sistemi ===")

    cam_index = int(input("Kamera indeksi (varsayılan 0): ").strip() or 0)
    cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        raise RuntimeError("Kamera açılamadı. Kamera indeksi veya erişim yetkisini kontrol edin.")

    state = AppState()
    frame_ref: dict[str, np.ndarray | None] = {"frame": None}
    buttons = UIButtons(
        yolluk_select=(15, 15, 180, 45),
        urun_select=(210, 15, 180, 45),
        color_select=(405, 15, 140, 45)
    )

    cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar("Beklenen Urun", WINDOW_NAME, state.expected_count, 200, _noop)
    cv2.createTrackbar("Verim Esigi (%)", WINDOW_NAME, state.threshold_percent, 100, _noop)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, {"state": state, "frame_ref": frame_ref, "buttons": buttons})

    print("\nKontroller:")
    print("- Üstteki butonlardan yolluk/ürün alanı ve renk seçimi yap")
    print("- 'q': Çıkış")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Hata] Kameradan görüntü alınamadı.")
            break

        frame_ref["frame"] = frame.copy()
        state.expected_count = max(cv2.getTrackbarPos("Beklenen Urun", WINDOW_NAME), 1)
        state.threshold_percent = cv2.getTrackbarPos("Verim Esigi (%)", WINDOW_NAME)

        vis = frame.copy()
        draw_ui_panel(vis, state, buttons)

        if state.start_point is not None and state.current_point is not None:
            cv2.rectangle(vis, state.start_point, state.current_point, (0, 165, 255), 2)

        urun_sayisi, yolluk_var, processed = count_products_and_yolluk(vis, state)

        if state.selected_bgr is not None:
            cv2.putText(processed, f"Secili Renk (BGR): {state.selected_bgr}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        minimum_required = state.expected_count * (state.threshold_percent / 100.0)
        signal = 1 if urun_sayisi < minimum_required else 0
        cv2.putText(processed, f"Beklenen: {state.expected_count} | Esik: %{state.threshold_percent}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(processed, f"Anlik urun: {urun_sayisi} | Cikis sinyali: {signal}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255) if signal == 1 else (0, 255, 0), 2)
        cv2.putText(processed, f"Yolluk: {'VAR' if yolluk_var else 'YOK'}", (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

        cv2.imshow(WINDOW_NAME, processed)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
