# Adaptive Exploration Strategies in Deep Reinforcement Learning for Task Offloading in Mobile Edge Computing under Non-Stationary Workloads

> **Riset Skripsi S1 Informatika**  
> **Topik:** Deep Reinforcement Learning (DRL) untuk Task Offloading pada Edge Computing dengan Workload Non-Stationary.  
> **Paper Fondasi Algoritmik:**  
> Ming Tang & Vincent W. S. Wong, *“Deep Reinforcement Learning for Task Offloading in Mobile Edge Computing Systems,”* **IEEE Transactions on Mobile Computing**, Vol. 21, No. 6, pp. 1985–1997, June 2022.  
> DOI: [10.1109/TMC.2020.3036871](https://doi.org/10.1109/TMC.2020.3036871) | arXiv: [2005.02459](https://arxiv.org/abs/2005.02459)

---

## Daftar Isi
1. [Ringkasan Penelitian](#1-ringkasan-penelitian)
2. [Struktur Direktori Repositori](#2-struktur-direktori-repositori)
3. [Perbedaan Implementasi (Legacy vs Paper-Aligned)](#3-perbedaan-implementasi-legacy-vs-paper-aligned)
4. [Instalasi dan Prasyarat](#4-instalasi-dan-prasyarat)
5. [Panduan Menjalankan Eksperimen (via .cfg)](#5-panduan-menjalankan-eksperimen-via-cfg)
6. [Preset Konfigurasi Siap Pakai](#6-preset-konfigurasi-siap-pakai)
7. [Pemanfaatan Docker (Dockerfile, Compose, & Helper Script)](#7-pemanfaatan-docker-dockerfile-compose--helper-script)
8. [Format File .cfg dan Parameter](#8-format-file-cfg-dan-parameter)
9. [Visualisasi dan Laporan Hasil](#9-visualisasi-dan-laporan-hasil)
10. [Sitasi](#10-sitasi)

---

## 1. Ringkasan Penelitian

Dalam arsitektur *Mobile Edge Computing* (MEC), sejumlah perangkat bergerak (*mobile devices*) harus mengambil keputusan sekuensial apakah komputasi tugas dieksekusi secara lokal atau di-offload ke salah satu server edge. Keputusan ini memengaruhi waktu tunggu (*queue waiting delay*), waktu transmisi nirkabel, dan penundaan komputasi (*execution delay*). Jika total waktu melampaui batas batas waktu (*deadline*), tugas akan didrop (*task dropping*).

Paper **Tang & Wong (2022)** mengusulkan algoritma **D3QN (Dueling Double Deep Q-Network)** yang diperkaya dengan lapisan **LSTM** untuk menangkap dinamika riwayat beban server (*historical load matrix* $\mathbf{H}(t)$). 

Namun, evaluasi pada paper Tang & Wong **hanya dilakukan pada kondisi lingkungan stasioner** ($p=0.3$ tetap). Penelitian skripsi ini berfokus pada:
1. **Evaluasi di Bawah Workload Non-Stasioner**: Menguji ketahanan D3QN ketika distribusi kedatangan tugas berubah (*sudden shift*, *sinusoidal*, *bursty*).
2. **Adaptasi Strategi Eksplorasi**: Meneliti pengaruh dan adaptasi probabilitas eksplorasi ($\epsilon$) saat terjadi pergeseran beban dinamis pada sistem edge.

---

## 2. Struktur Direktori Repositori

Repositori ini mendukung eksekusi multi-backend dengan struktur terorganisir:
1. **Root Directory (`./`)**: Kode *legacy* asli dari penulis paper (TensorFlow 1.x) yang telah diperbaiki bug antreannya dan didukung dual-backend (TensorFlow + PyTorch).
2. **`paper_aligned_code/`**: Implementasi baru yang **100% patuh secara matematis** terhadap persamaan dan formulasi di paper Tang & Wong (PyTorch, modular, konfigurasi `.cfg`, dan cost minimization $\arg\min$).

```
Deep-Q-learning-for-mobile-edge-computing-main/
│
├── Dockerfile                       # Environment multi-backend (TF 1.15.5 + PyTorch 1.13.1)
├── docker-compose.yml               # Orkestrasi container siap pakai (experiment, train, legacy)
├── run_docker.sh                    # Skrip bash pembantu (build, run, train, shell)
│
├── paper_aligned_code/              # [DIREKOMENDASIKAN] Implementasi Sesuai Paper
│   ├── config.cfg                   # File konfigurasi master
│   ├── config_parser.py             # Modul pembaca & validator .cfg
│   ├── configs/                     # Preset skenario eksperimen skripsi
│   │   ├── stationary.cfg           # Skenario beban stasioner (Tabel 1 paper)
│   │   ├── sinusoidal.cfg           # Skenario beban fluktuasi sinusoidal bertahap
│   │   ├── sudden_shift.cfg         # Skenario lonjakan beban mendadak (tanpa reset)
│   │   └── sudden_shift_reset.cfg   # Skenario lonjakan mendadak + Epsilon Reset (Seksi 5)
│   ├── mec_env.py                   # Environment MEC persis Eq. (1)-(18) & GPS Eq. (7)-(8)
│   ├── d3qn_network.py              # Dueling Network + LSTM persis Fig. 2 & Eq. (20)
│   ├── d3qn_agent.py                # Agen D3QN cost minimization (argmin) & decay epsilon
│   ├── workload.py                  # Generator tugas diskrit Lambda & pola non-stasioner
│   ├── train.py                     # Skrip training mandiri (mendukung --config)
│   ├── run_experiment.py            # Pipeline riset lengkap (Train + Baselines + Laporan)
│   ├── generate_report.py           # Generator 6 grafik publikasi 300 DPI
│   ├── reports/                     # Hasil grafik dan laporan markdown
│   └── README.md                    # Dokumentasi teknis mendalam paper_aligned_code
│
├── fog_env.py                       # Environment legacy (bugfix baris 332 diaplikasikan)
├── RL_brain.py                      # Brain legacy (Dual-Backend: PyTorch di host, TF di Docker)
├── workload.py                      # Generator workload untuk kode legacy
├── train.py                         # Skrip training kode legacy
├── run_experiment.py                # Pipeline eksperimen kode legacy
├── generate_report.py               # Generator grafik untuk kode legacy
├── reports/                         # Direktori laporan eksperimen kode legacy
├── DOKUMENTASI_SISTEM.md            # Laporan teknis mendalam & perbandingan kode
└── README.md                        # Panduan utama repositori ini
```

---

## 3. Perbedaan Implementasi (Legacy vs Paper-Aligned)

| Komponen | Implementasi Legacy (Root) | Implementasi Baru (`paper_aligned_code/`) | Keterangan Paper Tang & Wong |
| :--- | :--- | :--- | :--- |
| **Optimasi Aksi** | Reward Maximization ($\arg\max$) | **Cost Minimization ($\arg\min$)** | Sesuai Persamaan (19), (22), & (25) |
| **Definisi Epsilon** | Probabilitas eksploitasi ($0.0 \to 0.99$) | **Probabilitas eksplorasi ($1.0 \to 0.01$)** | Meluruh dari 1 ke 0.01 (Seksi 5) |
| **Learning Rate** | `0.01` (10x lebih agresif) | **`0.001`** | Sesuai Tabel 1 & Seksi 5 |
| **Batch Size** | `32` | **`16`** | Sesuai Seksi 5 & Catatan Kaki 7 |
| **Ukuran Task $\lambda_m$** | Kontinu `uniform(2.0, 5.0)` | **Diskrit $\Lambda = \{2.0, 2.1, \dots, 5.0\}$** | Sesuai Tabel 1 & Seksi 2.1.1 |
| **Buffer LSTM $\mathbf{H}(t)$** | Diiris sembarangan dari replay buffer | **Matriks time-series utuh $T^{\text{step}} \times N$** | Sesuai Seksi 3.1 & Persamaan (13) |
| **Antrean Transmisi** | Bug baris 332 menimpa antrean komputasi | **Terisolasi independen** | Sesuai Persamaan (5)–(6) |
| **Konfigurasi** | Hardcoded di Python script | **Didukung file eksternal `.cfg`** | Mempermudah eksperimen berulang |

---

## 4. Instalasi dan Prasyarat

### Opsi A: Menjalankan Langsung di Host (Linux / macOS / Windows)
Prasyarat sistem:
- Python 3.8+ (diuji pada Python 3.10 dan 3.12)
- Paket dependensi: PyTorch, NumPy, Matplotlib

```bash
pip install torch numpy matplotlib
```

*(Catatan: Repositori ini sudah dilengkapi direktori lokal `.venv_pkgs` untuk lingkungan Linux Ubuntu host).*

### Opsi B: Menggunakan Docker (Rekomendasi untuk Reproduktifitas)
Lihat [Bagian 7: Pemanfaatan Docker](#7-pemanfaatan-docker-dockerfile-compose--helper-script) untuk petunjuk lengkap membangun dan mengeksekusi container.

---

## 5. Panduan Menjalankan Eksperimen (via .cfg)

Gunakan skrip `run_experiment.py` di dalam `paper_aligned_code/` untuk menjalankan seluruh alur penelitian secara otomatis:
1. Melatih agen D3QN sesuai konfigurasi `.cfg`.
2. Mengevaluasi 4 baseline pembanding (*All-Local*, *Random Offloading*, *Greedy Queue-Aware*, dan *Proposed D3QN*).
3. Menghasilkan 6 visualisasi grafik berkualitas publikasi (300 DPI).
4. Menyusun laporan akhir berformat Markdown.

### Sintaks Dasar:
```bash
python3 paper_aligned_code/run_experiment.py --config <path/ke/file.cfg>
```

### Opsi Tambahan (CLI Override):
Argumen baris perintah akan menimpa (*override*) nilai yang ada di dalam file `.cfg` tanpa perlu mengedit file secara permanen:
```bash
# Menjalankan konfigurasi sudden_shift tetapi dengan 50 episode
python3 paper_aligned_code/run_experiment.py --config paper_aligned_code/configs/sudden_shift.cfg --episodes 50

# Mengubah jumlah perangkat mobile menjadi 30
python3 paper_aligned_code/run_experiment.py --config paper_aligned_code/configs/stationary.cfg --devices 30
```

---

## 6. Preset Konfigurasi Siap Pakai

Telah disediakan 4 preset file `.cfg` di direktori `paper_aligned_code/configs/`:

### 1. Skenario Stasioner (Baseline Paper Tang & Wong)
Kedatangan tugas bernilai konstan $p = 0.3$.
```bash
python3 paper_aligned_code/run_experiment.py --config paper_aligned_code/configs/stationary.cfg
```

### 2. Skenario Non-Stasioner Sinusoidal (Gradual Shift)
Probabilitas kedatangan tugas berubah secara periodik dan bertahap mengikuti gelombang sinus:
```bash
python3 paper_aligned_code/run_experiment.py --config paper_aligned_code/configs/sinusoidal.cfg
```

### 3. Skenario Sudden Shift (Tanpa Adaptasi Reset)
Beban kerja normal ($p=0.2$) melonjak tiba-tiba ke beban tinggi ($p=0.6$) di tengah episode, dengan agen D3QN standar yang sudah berada pada tahap eksploitasi ($\epsilon \approx 0.01$).
```bash
python3 paper_aligned_code/run_experiment.py --config paper_aligned_code/configs/sudden_shift.cfg
```

### 4. Skenario Sudden Shift dengan Epsilon Reset (Seksi 5 Paper)
Beban kerja melonjak tiba-tiba, namun agen mengaktifkan mekanisme adaptasi reset probabilitas eksplorasi ($\epsilon \to 1.0$) tepat saat terjadi pergeseran lingkungan.
```bash
python3 paper_aligned_code/run_experiment.py --config paper_aligned_code/configs/sudden_shift_reset.cfg
```

---

## 7. Pemanfaatan Docker (Dockerfile, Compose, & Helper Script)

Repositori ini dilengkapi dengan **Dockerfile multi-backend** yang mendukung:
- Lingkungan **TensorFlow 1.15.5** (Python 3.7) untuk kode legacy.
- Lingkungan **PyTorch 1.13.1 (CPU)** untuk implementasi paper-aligned dan parser `.cfg`.

### Cara 1: Menggunakan Helper Script `./run_docker.sh` (Paling Praktis)
Telah disediakan skrip helper [`run_docker.sh`](file://run_docker.sh):

```bash
# 1. Bangun image Docker (cukup sekali di awal)
./run_docker.sh build

# 2. Jalankan pipeline eksperimen dengan konfigurasi pilihan
./run_docker.sh run paper_aligned_code/configs/sudden_shift_reset.cfg
./run_docker.sh run paper_aligned_code/configs/stationary.cfg

# 3. Jalankan training saja
./run_docker.sh train paper_aligned_code/configs/sudden_shift.cfg

# 4. Jalankan kode legacy TensorFlow 1.15
./run_docker.sh legacy

# 5. Buka interactive shell di dalam container
./run_docker.sh shell
```

### Cara 2: Menggunakan Docker Compose
```bash
# Bangun dan jalankan pipeline eksperimen default
docker compose run --rm experiment

# Jalankan kode legacy
docker compose run --rm legacy

# Buka bash shell
docker compose run --rm shell
```

### Cara 3: Perintah `docker` Manual
```bash
# Build manual:
docker build -t d3qn-mec:latest .

# Run manual dengan volume mount (agar grafik & laporan tersimpan di host):
docker run --rm -it -v $(pwd):/app -w /app d3qn-mec:latest \
    python3 paper_aligned_code/run_experiment.py --config paper_aligned_code/configs/sudden_shift_reset.cfg
```

---

## 8. Format File .cfg dan Parameter

File konfigurasi terdiri dari empat blok utama:

```ini
[ENVIRONMENT]
num_devices = 20          # Jumlah mobile devices (M)
num_edge_nodes = 5        # Jumlah edge server nodes (N)
num_time_slots = 100      # Panjang episode dalam time slot (T)
max_deadline = 10         # Batas waktu tugas tau_m (time slot)
time_slot_duration = 0.1  # Durasi per time slot Delta (detik)
f_device = 2.5            # Kemampuan CPU mobile device (GHz)
f_edge = 41.8             # Total kemampuan CPU edge server (GHz)
r_tran = 14.0             # Data rate transmisi uplink (Mbps)
comp_density = 0.297      # Kebutuhan siklus CPU rho_m (gigacycles/Mbit)
penalty_cost = 20.0       # Penalti biaya jika task didrop

[DRL_AGENT]
learning_rate = 0.001     # Sesuai Tang & Wong (0.001)
discount_factor = 0.9     # Faktor diskon gamma
batch_size = 16           # Batch size mini-batch pengalaman
replace_threshold = 200   # Frekuensi pembaruan parameter Target Network
memory_capacity = 500     # Kapasitas memori Experience Replay
epsilon_init = 1.0        # Probabilitas eksplorasi awal
epsilon_min = 0.01        # Probabilitas eksplorasi minimal
n_lstm = 20               # Dimensi hidden state LSTM
n_l1 = 20                 # Dimensi lapisan fully-connected

[WORKLOAD]
pattern = sudden_shift    # Pilihan: stationary, sinusoidal, sudden_shift, bursty, markovian
base_arrival_prob = 0.3   # Probabilitas kedatangan dasar
min_size = 2.0            # Ukuran minimal task (Mbits)
max_size = 5.0            # Ukuran maksimal task (Mbits)
period = 50               # Periode osilasi (untuk pola sinusoidal)
seed = 42                 # Random seed reproduktifitas

[EXPERIMENT]
episodes = 50             # Total episode training
enable_epsilon_reset = true  # Aktifkan adaptasi eksplorasi reset
reset_at_episode = 25     # Titik episode dilakukannya reset
eval_episodes = 15        # Episode evaluasi performa kebijakan
output_dir = paper_aligned_code/reports # Lokasi output grafik & laporan
```

---

## 9. Visualisasi dan Laporan Hasil

Setiap kali skrip `run_experiment.py` selesai dieksekusi, sistem secara otomatis menghasilkan berkas visualisasi beresolusi 300 DPI di `paper_aligned_code/reports/figures/`:

1. **`learning_convergence.png`**: Grafik konvergensi nilai cost rata-rata dan peluruhan eksplorasi $\epsilon$ sepanjang episode training.
2. **`delay_evolution.png`**: Evolusi delay pemrosesan tugas sebelum vs sesudah perubahan workload.
3. **`drop_rate_breakdown.png`**: Persentase rasio tugas yang didrop (*task drop ratio*) beserta batas penalti.
4. **`offload_dynamics.png`**: Distribusi keputusan pemilihan aksi agen (eksekusi lokal vs offload ke masing-masing edge node 1–5).
5. **`baseline_comparison.png`**: Diagram perbandingan performa delay dan drop ratio antara D3QN usulan terhadap kebijakan *All-Local*, *Random*, dan *Greedy Queue-Aware*.
6. **`workload_comparison.png`**: Perbandingan komparatif lintas 4 pola beban kerja (*Stationary*, *Sinusoidal*, *Sudden Shift*, *Bursty*).

Laporan ringkasan naratif disimpan dalam format Markdown di:
- `paper_aligned_code/reports/FINAL_REPORT_PAPER_ALIGNED.md`

---

## 10. Sitasi

Jika Anda merujuk pada kode dan metodologi ini dalam naskah skripsi atau publikasi:

```bibtex
@article{tang2022deep,
  author    = {Tang, Ming and Wong, Vincent W. S.},
  journal   = {IEEE Transactions on Mobile Computing}, 
  title     = {Deep Reinforcement Learning for Task Offloading in Mobile Edge Computing Systems}, 
  year      = {2022},
  volume    = {21},
  number    = {6},
  pages     = {1985--1997},
  doi       = {10.1109/TMC.2020.3036871}
}
```
# DeepQ-MobileEdgeComputing_ThesisPurpose
