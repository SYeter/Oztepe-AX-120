# Enjeksiyon Makinesi Kamera Kontrol Uygulaması

Bu proje, enjeksiyon makinelerinde:
- **Yolluk var/yok kontrolü**
- **Ürün algılama ve sayım**
- **Verim eşiğine göre 1/0 sinyal üretimi**

işlemlerini **Python + OpenCV + PyQt5** ile gerçek zamanlı olarak yapar.

## Özellikler
- Modern, koyu temalı **PyQt5 masaüstü arayüzü**.
- Görüntü üzerinden sürükle-bırak ile **yolluk ROI** ve **ürün ROI** seçimi.
- Piksel tıklaması ile **renk seçimi**.
- Canlı metrik kartları:
  - Seçili renk
  - Anlık ürün sayısı
  - Çıkış sinyali (1/0)
  - Yolluk durumu
- Kullanıcının belirlediği **beklenen ürün adedi** ve **verim eşiği (%)** ile karar:
  - `ürün_sayısı < beklenen * (eşik/100)` ise **çıkış sinyali = 1**
  - aksi durumda **çıkış sinyali = 0**

## Kurulum
```bash
pip install -r requirements.txt
```

## Çalıştırma
```bash
python app.py
```

Program başlangıcında kamera indeksi ister (`0` varsayılan).

## Arayüz Kullanımı
1. **Yolluk Alanı Seç** veya **Ürün Alanı Seç** butonuna tıklayın.
2. Kamera görüntüsü üzerinde sürükleyip bırakarak ROI tanımlayın.
3. **Renk Seç** ile tespit edilecek rengi görüntüden seçin.
4. Beklenen adet ve eşik değerlerini girip **Değerleri Uygula** deyin.
5. Sonuçları soldaki canlı kartlardan takip edin.

## Notlar
- Aydınlatma değişimlerinde renk toleransı için `build_mask_by_selected_color` fonksiyonundaki `tol` değeri güncellenebilir.
- Kamera erişim sorunu varsa işletim sistemi izinlerini kontrol edin.
