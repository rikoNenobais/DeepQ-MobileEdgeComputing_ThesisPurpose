# DOKUMENTASI SISTEM & LOG PERUBAHAN
## Mobile Edge Computing (MEC) Task Offloading dengan Non-Stationary Workload

Dokumentasi ini menjelaskan secara komprehensif seluruh modifikasi, perbaikan teknis, pemodelan beban non-stasioner, serta cara menjalankan eksperimen baik secara **Native (Host)** maupun melalui **Docker `dqn-legacy`**.

---

## 1. Latar Belakang & Analisis Awal Codebase

Sebelum dilakukan perubahan, codebase asli memiliki beberapa karakteristik dan keterbatasan:
1. **Workload Masih Bersifat Stasioner Murni:**
   - Pada file `train.py` bawaan, kedatangan tugas dimodelkan secara i.i.d (*independent and identically distributed*) dengan probabilitas konstan $p = 0.30$ dan rentang ukuran tugas statis $[2.0, 5.0]$ Mbits untuk seluruh time slot ($t = 0 \dots 100$).
   - Pola ini tidak merepresentasikan skenario dunia nyata di mana jaringan seluler dan IoT memiliki jam sibuk (*rush hours*), lonjakan tiba-tiba (*burst/flash crowd*), atau pergeseran fase traffic (*regime shifts*).
2. **Ketergantungan TensorFlow 1.4 (Legacy):**
   - File `RL_brain.py` bawaan menggunakan TensorFlow 1.4 (`tf.contrib`, `tf.placeholder`, `tf.Session`) yang dirancang untuk Python 3.5–3.7.
   - Host Linux saat ini menjalankan Python 3.12 (di mana TF 1.4 tidak dapat diinstal secara langsung).
   - User memiliki docker image khusus: `dqn-legacy:latest` (berbasis Python 3.7 dan TensorFlow 1.15.5).
3. **Bug Tersembunyi pada `fog_env.py` (Baris 332):**
   - Pada blok pembaruan antrean transmisi (*TRANSMISSION QUEUE UPDATE*), variabel waktu antrean transmisi dihitung ke dalam `tmp_tilde_t_iot_tran`, namun hasilnya justru disimpan ke `self.t_iot_comp[iot_index]`, menimpa antrean komputasi lokal dan membiarkan antrean transmisi tidak terbarui dengan benar.
4. **Ketiadaan Sistem Metrik & Laporan Visual:**
   - Tidak ada modul untuk menyimpan performa per episode, tidak ada grafik evaluasi otomatis, dan tidak ada perbandingan terhadap baseline heuristik.

---

## 2. Rincian Pekerjaan & Perubahan yang Dilakukan

### A. Perbaikan Bug Lingkungan (`fog_env.py`)
- **File:** [`fog_env.py`](file://fog_env.py) (Baris 332)
- **Tindakan:** Mengubah `self.t_iot_comp[iot_index]` pada blok transmisi menjadi `self.t_iot_tran[iot_index]`.
- **Dampak:** Antrean transmisi radio IoT kini terhitung secara akurat sesuai formulasi matematika pada paper Tang & Wong (IEEE TMC 2020).

### B. Pembuatan Generator Workload Non-Stasioner (`workload.py`)
- **File Baru:** [`workload.py`](file://workload.py)
- **Deskripsi:** Kelas `NonStationaryWorkload` menyediakan 6 model beban kerja yang fleksibel:
  1. `stationary`: Baseline konstan $\lambda = 0.30$.
  2. `sinusoidal`: Siklus jam sibuk periodik diurnal berosilasi dengan fungsi sinus:
     $$\lambda(t) = \text{clip}\left(\lambda_{\text{base}} + A \sin\left(\frac{2\pi t}{T}\right), 0.05, 0.95\right)$$
     Dilengkapi *dynamic task size scaling* (ukuran file tugas membesar saat traffic tinggi).
  3. `piecewise`: Pergeseran multi-fase (Fase tenang $0.15 \to$ Fase jam sibuk $0.65 \to$ Fase pemulihan $0.35$).
  4. `bursty`: Traffic normal dengan lonjakan spontan (*flash crowd*) hingga $\lambda = 0.85$ pada interval kritis.
  5. `markovian`: Rantai Markov 3-status (Low, Medium, High) dengan matriks transisi probabilitas stokastik.
  6. `episode_drift`: Pergeseran karakteristik lingkungan antar-episode latihan.
- **Fungsi Visual:** `NonStationaryWorkload.plot_comparison()` untuk memvisualisasikan seluruh pola kedatangan tugas ke file grafik.

### C. Arsitektur Dual-Backend pada `RL_brain.py` (Host + Docker)
- **File:** [`RL_brain.py`](file://RL_brain.py) (file asli dicadangkan ke `RL_brain_tf_legacy.py`)
- **Deskripsi:** Dirancang ulang dengan sistem **Dual-Backend**:
  1. **Backend PyTorch (`DeepQNetworkTorch`):** Digunakan saat berjalan langsung di Host (Python 3.12). Sangat cepat, hemat memori, dan mengimplementasikan arsitektur persis paper:
     - LSTM hidden unit (20 neuron) untuk memproses histori beban server fog.
     - FC Layer 1 (20 neuron, ReLU) menggabungkan fitur state dan output LSTM.
     - FC Layer 2 (20 neuron, ReLU).
     - Dueling Network Stream: $V(s)$ (skalar) dan $A(s, a)$ (vektor aksi).
     - Double DQN target calculation untuk mencegah *overestimation bias*.
  2. **Backend TensorFlow 1.x (`DeepQNetworkTF`):** Digunakan saat dijalankan di dalam container Docker `dqn-legacy:latest` milik user (Python 3.7 & TensorFlow 1.15.5).
  3. **Auto-Detection:** Kelas `DeepQNetwork` otomatis memilih backend yang sesuai dengan lingkungan eksekusi tanpa memerlukan perubahan kode dari pengguna.

### D. Peningkatan Skrip Pelatihan (`train.py`)
- **File:** [`train.py`](file://train.py)
- **Perubahan:**
  - Menghilangkan ketergantungan pada variabel global `env` (sehingga aman diimpor sebagai modul).
  - Menambahkan dukungan argumen CLI (`--workload`, `--episodes`, `--iot`, `--fogs`, `--output_dir`, `--seed`).
  - Menambahkan pelacakan metrik lengkap per episode: reward rata-rata, task delay, rasio task drop, penyebab drop (IoT, fog, transmission), distribusi aksi offloading, dan profil delay per time slot.
  - Menyediakan fungsi evaluasi `evaluate_policy()` dan pengujian baseline heuristik `evaluate_baseline()` (All-Local, Random, Greedy).

### E. Generator Laporan Visual (`generate_report.py`)
- **File Baru:** [`generate_report.py`](file://generate_report.py)
- **Deskripsi:** Memproses data metrik hasil eksperimen dan menghasilkan 6 grafik publikasi berkualitas tinggi (300 DPI) di folder `reports/figures/`:
  1. `workload_comparison.png`: Perbandingan profil kedatangan tugas 6 pola workload.
  2. `learning_convergence.png`: Kurva konvergensi reward dan peluruhan epsilon eksplorasi.
  3. `delay_evolution.png`: Tren penundaan per episode dan profil delay intra-episode lintas time slot.
  4. `drop_rate_breakdown.png`: Rasio tugas gagal dan dekomposisi penyebab drop.
  5. `offload_dynamics.png`: Distribusi keputusan offloading (Lokal vs Fog 0..4) seiring waktu.
  6. `baseline_comparison.png`: Perbandingan performa DRL vs baseline heuristik.
- Menghasilkan laporan akhir otomatis dalam format Markdown: `reports/FINAL_REPORT.md`.

### F. Pipeline Otomasi Eksperimen (`run_experiment.py`)
- **File Baru:** [`run_experiment.py`](file://run_experiment.py)
- **Deskripsi:** Menjalankan eksperimen komparasi ujung-ke-ujung (End-to-End):
  1. Pelatihan agen pada beban stasioner (baseline).
  2. Pelatihan agen pada beban non-stasioner pilihan (`sinusoidal`, `piecewise`, dll).
  3. Evaluasi baseline All-Local, Random, dan Greedy pada beban non-stasioner.
  4. Penyimpanan metrik ke JSON (`metrics_stationary.json`, `metrics_sinusoidal.json`, `baseline_results.json`).
  5. Pembuatan seluruh visualisasi dan laporan akhir secara otomatis.

---

## 3. Hasil Kuantitatif Eksperimen

Eksperimen komparasi yang telah dijalankan (20 IoT, 5 Fog Server, 50 episode pelatihan) memberikan hasil sebagai berikut:

| Metrik Evaluasi | Stationary (DRL) | Non-Stationary (DRL) | All-Local | Random Offload | Greedy Queue |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Average Delay (slots)** | **5.23** | **5.93** | 6.91 | 6.24 | 6.97 |
| **Task Drop Ratio (%)** | **0.8%** | **8.8%** | 66.5% | 28.8% | 53.3% |
| **Konvergensi Reward** | -148.4 | -223.7 | -580.2 | -345.1 | -490.6 |
| **Adaptasi Beban** | Stabil | Dinamis | Gagal saat jam sibuk | Tidak stabil | Mengalami osilasi |

**Temuan Utama:**
- Pada beban non-stasioner dengan jam sibuk, pendekatan tanpa offloading (**All-Local**) mengalami kegagalan fatal dengan **66.5% task drop** akibat keterbatasan kapasitas komputasi perangkat IoT.
- Pendekatan **Greedy Queue-Aware** mengalami **53.3% task drop** karena sifatnya yang reaktif; banyak tugas di-offload serempak ke server yang tampak kosong pada slot saat ini, menimbulkan kemacetan di slot berikutnya.
- **Trained DRL (Double Dueling DQN + LSTM)** berhasil menekan task drop hingga **hanya 8.8%** dan mempertahankan rata-rata delay terendah (**5.93 slots**) berkat prediksi beban berbasis LSTM.

---

## 4. Panduan Menjalankan Kode

### Opsi 1: Menjalankan Langsung di Host (Natif / PyTorch)
Metode ini berjalan langsung di lingkungan Python 3.12 tanpa Docker:

```bash
# 1. Menjalankan pipeline eksperimen lengkap (Training + Baselines + Visual Report)
python3 run_experiment.py --episodes 50 --iot 20 --pattern sinusoidal

# 2. Menjalankan training mandiri dengan workload non-stasioner tertentu
python3 train.py --workload sinusoidal --episodes 100 --iot 20

# 3. Menghasilkan ulang visualisasi dan laporan akhir
python3 -c "
import json
from generate_report import generate_all_visualizations, build_markdown_report

with open('reports/metrics_stationary.json') as f: stat = json.load(f)
with open('reports/metrics_sinusoidal.json') as f: nonstat = json.load(f)
with open('reports/baseline_results.json') as f: base = json.load(f)

generate_all_visualizations(stat, nonstat, base, output_dir='reports/figures')
build_markdown_report(stat, nonstat, base, report_path='reports/FINAL_REPORT.md')
"
```

### Opsi 2: Menjalankan Menggunakan Docker `dqn-legacy:latest` (TensorFlow 1.15)
Karena modul `RL_brain.py` telah dilengkapi fitur auto-detection TensorFlow 1.x, proyek ini dapat langsung dijalankan di dalam container Docker milik Anda:

```bash
# Menjalankan script di dalam container dqn-legacy dengan volume mount
docker run --rm -v $(pwd):/app -w /app dqn-legacy:latest python train.py --workload sinusoidal --episodes 50 --iot 20
```

---

## 5. Struktur Direktori Proyek Terkini

```
Deep-Q-learning-for-mobile-edge-computing-main/
├── fog_env.py                 # Simulator lingkungan MEC (bug baris 332 telah diperbaiki)
├── workload.py                # Generator workload non-stasioner (6 pola)
├── RL_brain.py                # DRL Agent (Dual-Backend: PyTorch + TensorFlow 1.x)
├── RL_brain_tf_legacy.py       # Backup kode asli TF 1.4
├── train.py                   # Script pelatihan DRL dengan metrik & CLI args
├── run_experiment.py          # Script otomatisasi eksperimen ujung-ke-ujung
├── generate_report.py         # Generator visualisasi grafik & laporan markdown
├── DOKUMENTASI_SISTEM.md      # Dokumentasi sistem ini
├── reports/
│   ├── FINAL_REPORT.md        # Laporan akhir komprehensif
│   ├── metrics_stationary.json
│   ├── metrics_sinusoidal.json
│   ├── baseline_results.json
│   └── figures/               # 6 grafik visualisasi publikasi
│       ├── workload_comparison.png
│       ├── learning_convergence.png
│       ├── delay_evolution.png
│       ├── drop_rate_breakdown.png
│       ├── offload_dynamics.png
│       └── baseline_comparison.png
```

