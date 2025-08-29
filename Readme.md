# Note development

### **AUC Score from jurnal train 97,94%, val 97,31%**

- menggunakan randomseed selama pengembangan. Setiap kali Anda mengubah arsitektur, hyperparameter, atau memperbaiki kode, perbedaan hasil murni disebabkan oleh perubahan tersebut, bukan karena hoki.
- saat evaluasi akhir menggunakan 5-10 seed berbeda, dan laporkan mean dan standard deviation untuk membuktikan bahwa performa tersebut **stabil, robust, dan bukan sekadar kebetulan**
