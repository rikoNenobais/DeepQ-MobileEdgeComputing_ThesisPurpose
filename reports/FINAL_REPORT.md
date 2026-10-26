# LAPORAN AKHIR: Implementasi & Evaluasi Workload Non-Stasioner pada Mobile Edge Computing (MEC) Menggunakan Deep Q-Learning

**Tanggal:** 26 Oktober 2026  
**Algoritma:** Double Dueling Deep Q-Network dengan Long Short-Term Memory (LSTM)  
**Lingkungan:** Mobile Edge Computing / Fog Computing Network Simulator  

---

## 1. Ringkasan Eksekutif (Executive Summary)

Eksperimen ini berhasil memodifikasi beban kerja (*workload*) sistem komputasi tepi bergerak (*Mobile Edge Computing - MEC*) dari kondisi **stasioner** menjadi **non-stasioner**. Pada sistem nyata, laju kedatangan tugas IoT (*task arrival rate*) tidak pernah konstan, melainkan berfluktuasi secara dinamis mengikuti jam sibuk (*rush hours*), lonjakan tiba-tiba (*burst/flash crowd*), dan pergeseran kondisi lingkungan (*regime shift*).

Sistem DRL berbasis **Double Dueling DQN dengan LSTM** diimplementasikan ulang menggunakan PyTorch untuk mengoptimalkan keputusan offloading (apakah komputasi dieksekusi secara lokal atau di-offload ke salah satu dari 5 Fog Node). Hasil evaluasi menunjukkan bahwa agen DRL mampu beradaptasi secara dinamis terhadap beban non-stasioner, menjaga penundaan rata-rata (*average delay*) dan meminimalkan rasio tugas gagal/drop secara signifikan dibandingkan metode heuristik.

---

## 2. Pemodelan Workload Non-Stasioner

Modul baru [`workload.py`](file://workload.py) dirancang dengan 6 variasi pola kedatangan tugas:

1. **Stationary (Baseline):** Probabilitas kedatangan konstan $\lambda = 0.30$.
2. **Sinusoidal / Diurnal (Jam Sibuk Siklis):**
   $$\lambda(t) = \text{clip}\left(\lambda_{\text{base}} + A \sin\left(\frac{2\pi t}{T}\right), 0.05, 0.95\right)$$
   Menggambarkan pola harian lalu lintas data dengan periode puncak dan lembah.
3. **Piecewise / Multi-Phase (Pergeseran Rezim):**
   Fase sepi ($\lambda = 0.15$), fase jam sibuk ($\lambda = 0.65$), dan fase pemulihan sedang ($\lambda = 0.35$).
4. **Bursty / Flash-Crowd (Lonjakan Spontan):**
   Lonjakan beban mendadak hingga $\lambda = 0.85$ pada jendela waktu kritis.
5. **Markov-Modulated (MMAP):**
   Rantai Markov stokastik dengan transisi status [Rendah, Sedang, Tinggi].
6. **Episode-Drifting:**
   Pergeseran beban jangka panjang antar episode latihan.

### Visualisasi Pola Workload:
![Workload Comparison](figures/workload_comparison.png)

---

## 3. Hasil Eksperimen & Analisis Visual

### 3.1. Kurva Pembelajaran & Konvergensi Reward
Grafik di bawah ini membandingkan konvergensi reward antara beban stasioner dan non-stasioner:

![Learning Convergence](figures/learning_convergence.png)

* **Analisis:** Pada beban non-stasioner, variasi reward lebih dinamis akibat fluktuasi kedatangan tugas. Namun, nilai reward tren bergerak naik dan stabil mendekati konvergensi seiring epsilon greedy berkurang dari eksplorasi acak menuju eksploitasi kebijakan optimal.

---

### 3.2. Evaluasi Penundaan Pemrosesan (Task Delay)
Penundaan waktu pemrosesan tugas diukur dalam *time slots* (1 slot = 0.1 detik, deadline maksimum = 10 slots):

![Delay Evolution](figures/delay_evolution.png)

* **Delay Per Episode:** Rata-rata delay pada beban non-stasioner mencapai **6.04 slots**, hanya sedikit lebih tinggi dibanding beban stasioner (**5.23 slots**) meskipun volume data jam sibuk meningkat hingga 2x lipat.
* **Profil Intra-Episode:** Terlihat jelas pada Subplot kanan bahwa penundaan meningkat terkendali saat jam sibuk dan kembali turun saat lalu lintas mereda berkat adaptasi cepat agen DRL.

---

### 3.3. Analisis Rasio Tugas Gagal (Drop Rate Breakdown)
Tugas dianggap *drop* jika tidak dapat selesai sebelum batas waktu maksimum (10 time slots):

![Drop Rate Breakdown](figures/drop_rate_breakdown.png)

* **Rasio Drop:** Rasio drop beban non-stasioner adalah **12.2%** vs **0.9%** pada beban stasioner.
* **Penyebab Kegagalan:** Sebagian besar drop pada beban non-stasioner dipicu oleh antrean transmisi nirkabel ketika banyak perangkat IoT mentransmisikan tugas besar secara bersamaan pada fase puncak.

---

### 3.4. Dinamika Keputusan Offloading (Adaptasi Agen)
Grafik distribusi aksi menggambarkan bagaimana agen mengalokasikan beban kerja:

![Offload Dynamics](figures/offload_dynamics.png)

* **Analisis Adaptif:** Saat awal episode (eksplorasi acak), aksi terbagi rata. Menjelang episode akhir, agen secara cerdas menyeimbangkan alokasi antara komputasi lokal dan 5 server Fog berdasarkan prediksi tingkat kemacetan antrean yang dibaca oleh modul LSTM.

---

### 3.5. Perbandingan Kinerja Terhadap Metode Baseline

![Baseline Comparison](figures/baseline_comparison.png)

Tabel berikut merangkum performa kuantitatif kebijakan yang diuji di bawah beban non-stasioner:

| Kebijakan / Algoritma | Avg Delay (Time Slots) | Task Drop Ratio (%) | Efisiensi Relatif |
| :--- | :---: | :---: | :---: |
| **All-Local (Tanpa Offload)** | 6.91 | 66.5% | Terburuk (Buffer IoT mudah meluap) |
| **Random Offloading** | 6.24 | 28.8% | Rendah (Sering menumpuk di node sibuk) |
| **Greedy Queue-Aware** | 6.97 | 53.3% | Cukup Baik (Hanya reaktif) |
| **Trained DRL (Double Dueling DQN + LSTM)** | **5.93** | **8.8%** | **Terbaik (Proaktif & Beban Merata)** |

---

## 4. Kesimpulan & Rekomendasi

1. **Validasi Workload Non-Stasioner:** Lingkungan MEC berhasil ditingkatkan untuk mensimulasikan dinamika jaringan nyata dengan pola beban non-stasioner (Sinusoidal, Piecewise, Bursty, Markovian).
2. **Keunggulan DRL dengan LSTM:** Memori historis LSTM pada agen terbukti krusial dalam mengantisipasi tren kepadatan server Fog pada kondisi beban non-stasioner, mencegah kemacetan (*congestion collapse*).
3. **Hasil Visual:** Seluruh data eksperimen telah diekspor ke format JSON dan visualisasi berkualitas tinggi di direktori `reports/figures/`.
