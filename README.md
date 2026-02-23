# Enjeksiyon Makinesi Kamera Kontrol Uygulaması

Bu proje, enjeksiyon makinelerinde:
- **Yolluk var/yok kontrolü**
- **Ürün algılama ve sayım**
- **Verim eşiğine göre 1/0 sinyal üretimi**

işlemlerini Python + OpenCV ile gerçek zamanlı olarak yapar.

## Özellikler
- Mouse ile kullanıcı tanımlı **yolluk ROI** ve **ürün ROI** alanları.
- Ekrandan tıklayarak **algılanacak rengin seçilmesi**.
- Renk tabanlı görüntü işleme ile ürün sayımı.
- Kullanıcının belirlediği **beklenen ürün adedi** ve **verim eşiği (%)** ile karar:
  - `ürün_sayısı < beklenen * (eşik/100)` ise **çıkış sinyali = 1**
  - aksi durumda **çıkış sinyali = 0**

## Kurulum (PyCharm ile uyumlu)
1. Projeyi PyCharm ile açın.
2. `Python 3.10+` interpreter seçin.
3. Terminalden bağımlılıkları kurun:

```bash
pip install -r requirements.txt
```

## Çalıştırma
```bash
python app.py
```

Program başlangıcında sizden şunları ister:
- Kalıptan çıkması gereken ürün sayısı
- Verim eşiği (%)
- Kamera indeksi (genelde `0`)

## Klavye ve Mouse Kontrolleri
- `y` : Yolluk alanı (ROI) seçimini başlatır
- `u` : Ürün algılama alanı (ROI) seçimini başlatır
- **Sol tık (ROI modu kapalıyken)** : Algılanacak rengi seçer
- `q` : Uygulamadan çıkar

## Notlar
- Farklı aydınlatma koşullarında renk toleransı için `app.py` içindeki `build_mask_by_selected_color` fonksiyonundaki tolerans değeri ayarlanabilir.
- Kameraya erişim sorunu varsa işletim sistemi kamera izinlerini kontrol edin.
