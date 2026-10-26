# Implementasi D3QN MEC Task Offloading (Strict Paper-Conforming)

Folder `paper_aligned_code/` ini berisi implementasi kode baru yang **sepenuhnya disesuaikan dan dikonformasikan dengan isi paper referensi**:

> **Ming Tang & Vincent W.S. Wong**, *“Deep Reinforcement Learning for Task Offloading in Mobile Edge Computing Systems,”* **IEEE Transactions on Mobile Computing**, Vol. 21, No. 6, pp. 1985–1997, June 2022.  
> DOI: [10.1109/TMC.2020.3036871](https://doi.org/10.1109/TMC.2020.3036871) | arXiv: [2005.02459](https://arxiv.org/abs/2005.02459)

---

## 1. Pemetaan Persamaan Matematis Paper ke Kode

| Bagian Paper | Formulasi Matematika di Paper | Implementasi di `paper_aligned_code/` |
| :--- | :--- | :--- |
| **Seksi 2.1.1 (Ukuran Task)** | Set Diskrit $\Lambda = \{2.0, 2.1, \dots, 5.0\}$ Mbits | `self.task_size_set = np.round(np.arange(2.0, 5.05, 0.1), 1)` ([`mec_env.py`](file://mec_env.py)) |
| **Seksi 2.1.3 (Antrean Komputasi)** | Eq. (2) $\delta_m^{\text{comp}}$ & Eq. (3) $l_m^{\text{comp}}$ | `_get_waiting_time_comp()` & update antrean FIFO ([`mec_env.py`](file://mec_env.py)) |
| **Seksi 2.1.4 (Antrean Transmisi)** | Eq. (5) $\delta_m^{\text{tran}}$ & Eq. (6) $l_m^{\text{tran}}$ | Diperbaiki, antrean transmisi tidak lagi menimpa `t_iot_comp` ([`mec_env.py`](file://mec_env.py)) |
| **Seksi 2.2.1 (Edge GPS Sharing)** | Eq. (7) $B_n(t)$ & Eq. (8) Alokasi Kapasitas $\frac{f_n^{\text{edge}} \Delta}{\rho_m B_n(t)}$ | Generalized Processor Sharing (GPS) dihitung per time slot $t$ ([`mec_env.py`](file://mec_env.py)) |
| **Seksi 3.1 (State Space)** | Eq. (13) $s_m(t) = (\lambda_m(t), \delta^{\text{comp}}, \delta^{\text{tran}}, \mathbf{q}^{\text{edge}}, \mathbf{H}(t))$ | State lokal digabung dengan matriks $\mathbf{H}(t)$ utuh ($T^{\text{step}} \times N$) ([`mec_env.py`](file://mec_env.py)) |
| **Seksi 3.3 (Cost Function)** | Eq. (17) $c_m = \text{Delay}_m$, Eq. (18) $c_m = C$ (penalti drop) | $c_m(t) \ge 1$ dan penalti $C = 20$, bukan reward negatif ([`mec_env.py`](file://mec_env.py)) |
| **Seksi 4.1 (Arsitektur Jaringan)** | Fig. 2 & Fig. 3: Input $\to$ LSTM $\to$ 2 FC (ReLU) $\to$ A&V $\to$ Q | [`d3qn_network.py`](file://d3qn_network.py) (`D3QNPaperNet`) |
| **Seksi 4.1.4 (Dueling DQN)** | Eq. (20) $Q = V + (A - \frac{1}{\|\mathcal{A}\|} \sum A)$ | `Q = V + (A - A.mean(dim=1, keepdim=True))` ([`d3qn_network.py`](file://d3qn_network.py)) |
| **Seksi 4.2.1 (Seleksi Aksi)** | Eq. (22) $a_m(t) = \arg\min_a Q(s, a)$ (Cost Minimization) | `torch.argmin(q_values, dim=1)` saat eksploitasi ([`d3qn_agent.py`](file://d3qn_agent.py)) |
| **Seksi 4.2.2 (Double DQN Target)** | Eq. (25) $a^{\text{Next}} = \arg\min Q(s', a)$ & Eq. (26) $\hat{Q} = c + \gamma Q(s', a^{\text{Next}})$ | Seleksi aksi $\arg\min$ oleh Eval Net, evaluasi nilai oleh Target Net ([`d3qn_agent.py`](file://d3qn_agent.py)) |
| **Seksi 5 (Jadwal Epsilon)** | Probabilitas eksplorasi acak $\epsilon$ meluruh dari $1.0 \to 0.01$ | `self.epsilon = 1.0` decaying ke $0.01$ ([`d3qn_agent.py`](file://d3qn_agent.py)) |
| **Seksi 5 (Reset Epsilon)** | Reset eksplorasi saat lingkungan berubah non-stasioner | Metode `reset_exploration(1.0)` ([`d3qn_agent.py`](file://d3qn_agent.py)) |

---

## 2. Struktur File dalam Folder `paper_aligned_code/`

```
paper_aligned_code/
├── config.cfg            # Master config file (format INI standar)
├── config_parser.py      # Parser konfigurasi mandiri berbasis modul configparser
├── configs/              # Preset skenario riset skripsi:
│   ├── stationary.cfg           # Skenario stationary baseline
│   ├── sinusoidal.cfg           # Skenario non-stationary sinusoidal (gradual)
│   ├── sudden_shift.cfg         # Skenario sudden shift tanpa adaptasi
│   └── sudden_shift_reset.cfg   # Skenario sudden shift dengan Tang & Wong epsilon reset
├── mec_env.py            # Simulator lingkungan MEC persis Eq. (1)-(18)
├── d3qn_network.py       # Neural network D3QN persis Fig. 2 & Eq. (20)
├── d3qn_agent.py         # Agen DRL persis Algoritma 1 & 2 dengan argmin cost minimization
├── workload.py           # Generator workload dengan set diskrit Lambda & pola non-stasioner
├── train.py              # Skrip pelatihan terdistribusi (mendukung --config)
├── run_experiment.py     # Pipeline riset terpadu (Training + Baselines + Figures + Report)
├── generate_report.py    # Generator visualisasi grafik 300 DPI & laporan markdown
├── reports/              # Folder output hasil grafik PNG & laporan analisis
└── README.md             # Panduan teknis folder ini
```

---

## 3. Menjalankan Eksperimen Menggunakan File `.cfg`

### 3.1 Menjalankan Pipeline Riset Terpadu (Sangat Direkomendasikan)
Skrip `run_experiment.py` akan melatih model, mengevaluasi 4 baseline (*All-Local*, *Random*, *Greedy Queue-Aware*, dan *D3QN Proposed*), membuat 6 grafik visualisasi 300 DPI, dan menyusun laporan akhir.

```bash
# Menjalankan skenario sudden shift dengan adaptasi reset eksplorasi
python3 paper_aligned_code/run_experiment.py --config paper_aligned_code/configs/sudden_shift_reset.cfg

# Menjalankan skenario stasioner murni (baseline paper)
python3 paper_aligned_code/run_experiment.py --config paper_aligned_code/configs/stationary.cfg

# Menjalankan skenario non-stasioner sinusoidal
python3 paper_aligned_code/run_experiment.py --config paper_aligned_code/configs/sinusoidal.cfg
```

### 3.2 Menjalankan Training Mandiri
Jika hanya ingin melatih agen D3QN tanpa benchmarking baseline:
```bash
python3 paper_aligned_code/train.py --config paper_aligned_code/configs/sudden_shift.cfg
```

### 3.3 Override Parameter melalui CLI
Anda dapat menimpa parameter tertentu di file `.cfg` secara langsung tanpa mengubah file konfigurasi:
```bash
python3 paper_aligned_code/run_experiment.py --config paper_aligned_code/configs/sudden_shift.cfg --episodes 100 --devices 30
```

### 3.4 Menjalankan Melalui Docker Container
Dari root folder proyek, Anda dapat memanfaatkan container terisolasi:
```bash
./run_docker.sh run paper_aligned_code/configs/sudden_shift_reset.cfg
```

---

## 4. Parameter File Konfigurasi

| Seksi | Parameter | Default | Keterangan |
| :--- | :--- | :--- | :--- |
| `[ENVIRONMENT]` | `num_devices` | `20` | Jumlah mobile devices ($M$) |
| | `num_edge_nodes` | `5` | Jumlah edge nodes ($N$) |
| | `num_time_slots` | `100` | Jumlah time slot per episode ($T$) |
| | `max_deadline` | `10` | Batas waktu $\tau_m$ (slot) |
| | `time_slot_duration` | `0.1` | Durasi slot $\Delta$ (detik) |
| | `f_device` | `2.5` | Frekuensi CPU perangkat (GHz) |
| | `f_edge` | `41.8` | Frekuensi CPU edge node (GHz) |
| | `r_tran` | `14.0` | Kecepatan uplink (Mbps) |
| | `comp_density` | `0.297` | Kebutuhan komputasi $\rho_m$ (Gcycles/Mbit) |
| | `penalty_cost` | `20.0` | Penalti cost jika tugas didrop ($C$) |
| `[DRL_AGENT]` | `learning_rate` | `0.001` | Learning rate optimizer ($\alpha$) |
| | `discount_factor` | `0.9` | Discount factor ($\gamma$) |
| | `batch_size` | `16` | Ukuran mini-batch $|I|$ |
| | `replace_threshold` | `200` | Frekuensi update Target Net |
| | `memory_capacity` | `500` | Ukuran Experience Replay |
| | `epsilon_init` | `1.0` | Nilai eksplorasi awal ($\epsilon$) |
| | `epsilon_min` | `0.01` | Nilai eksplorasi minimum |
| | `n_lstm` | `20` | Hidden unit LSTM |
| | `n_l1` | `20` | Hidden unit Fully Connected |
| `[WORKLOAD]` | `pattern` | `sudden_shift` | Pola kedatangan: `stationary`, `sinusoidal`, `sudden_shift`, `bursty` |
| | `base_arrival_prob` | `0.3` | Probabilitas kedatangan dasar ($p$) |
| | `min_size` | `2.0` | Ukuran task minimum (Mbit) |
| | `max_size` | `5.0` | Ukuran task maksimum (Mbit) |
| | `period` | `50` | Periode osilasi gelombang beban |
| `[EXPERIMENT]` | `episodes` | `50` | Jumlah episode pelatihan |
| | `enable_epsilon_reset` | `true` | Mekanisme reset eksplorasi $\epsilon \to 1.0$ |
| | `reset_at_episode` | `25` | Titik episode reset dilakukan |
| | `eval_episodes` | `15` | Episode evaluasi performa akhir |
| | `output_dir` | `paper_aligned_code/reports` | Direktori penyimpanan grafik |
