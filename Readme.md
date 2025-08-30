# Note development

### **AUC Score from jurnal train 97,94%, val 97,31%**

- menggunakan randomseed selama pengembangan. Setiap kali mengubah arsitektur, hyperparameter, atau memperbaiki kode, perbedaan hasil disebabkan oleh perubahan tersebut.
- saat evaluasi akhir menggunakan 5-10 seed berbeda, dan mencatat mean dan standard deviation untuk membuktikan bahwa performa **stabil, robust, dan bukan sekadar kebetulan**

---

## 1. data_check

logic untuk memeriksa **konsistensi** dari setiap input. Dalam data geospasial (`.tif`) memiliki atribut lebih dari sekedar gambar. atributnya adalah:

1. Atribut spasial
   - Dimensi raster: width dan height dalam satuan pixel.
   - Sistem Referensi Koordinat: Mendefinisikan bagaimana grid 2D dari pixel dipetakan ke permukaan bumi yang melengkung.
   - GeoTransform: Enam angka yang mendefinisikan
     1. Koordinat X dari pojok kiri atas piksel pertama.
     2. Lebar satu piksel dalam satuan peta.
     3. Rotasi baris.
     4. Koordinat Y dari pojok kiri atas piksel pertama.
     5. Rotasi kolom.
     6. Tinggi satu piksel dalam satuan peta
2. Atribut data
   - Jumlah band: berapa banyak lapisan data yang ada dalam satu file.
   - Tipe data: Jenis nilai yang disimpan di setiap piksel
3. Metadata
   - Informasi tambahan seperti tanggal akuisisi data, sumber data, unit pengukuran, dll.

Konsistensi yang di periksa pada logic ini adalah `weight`, `height`, dan `GeoTransform`.

---
