# Notebook Google Colab — Python Dasar sampai Machine Learning

**Materi persiapan Ekshibisi Kecerdasan Artifisial · LKS Dikmen Tingkat Nasional 2026**
Sasaran: siswa SMKN Bali Mandara, Program Keahlian Teknik Komputer dan Jaringan (TKJ)

Empat notebook berurutan, dari belum pernah menulis Python sampai selesai
membangun dan menguji dua model machine learning. Seluruhnya berjalan di Google
Colab — tidak ada yang perlu dipasang di laptop.

---

## Isi folder

| Berkas | Isi | Waktu | Prasyarat |
|---|---|---|---|
| `01_python_dasar.ipynb` | variabel, tipe data, `if`, `for`, `list`/`dict`, fungsi, baca CSV, baca pesan galat | 90–120 menit | tidak ada |
| `02_numpy_pandas_eda.ipynb` | NumPy, pandas, muat dataset dari internet, delapan pertanyaan EDA, lima jenis grafik | 90–120 menit | notebook 01 |
| `03_klasifikasi_penguins.ipynb` | **studi kasus klasifikasi** — alur penuh: bagi data, baseline, Pipeline, empat algoritma, cross-validation, confusion matrix, tuning, uji akhir, simpan model, Responsible AI | 120–150 menit | notebook 01–02 |
| `04_regresi_california_housing.ipynb` | **studi kasus regresi** — alur yang sama dengan target berupa angka: MAE/RMSE/R², diagnosis residual, rekayasa fitur, metrik per kelompok, Responsible AI | 120–150 menit | notebook 01–03 |

Total sekitar 7–9 jam kalau semua latihan dikerjakan. Dirancang untuk dipakai
bertahap, bukan sekali duduk.

---

## Cara membuka di Google Colab

**Yang dibutuhkan:** akun Google, peramban, dan koneksi internet. Tidak perlu
memasang Python, Anaconda, atau apa pun.

1. Buka [colab.research.google.com](https://colab.research.google.com).
2. Menu **File → Upload notebook**.
3. Pilih berkas `.ipynb` yang diinginkan, atau tarik berkasnya ke jendela Colab.
4. Klik sel kode, tekan **Shift + Enter** untuk menjalankan.
5. **Jalankan dari atas ke bawah secara berurutan.** Sel bawah memakai variabel
   dari sel atas; melompat menghasilkan galat `NameError`.
6. Kalau kacau: **Runtime → Restart session**, lalu **Runtime → Run all**.

Notebook yang sudah diunggah tersimpan otomatis di Google Drive pemiliknya,
folder `Colab Notebooks`. Setiap siswa mengerjakan salinannya sendiri.

### Cara lain

- **Dari Google Drive** — unggah keempat berkas ke Drive, lalu klik ganda,
  pilih *Open with → Google Colaboratory*. Paling praktis untuk kelas: guru
  membagikan satu folder Drive, siswa memilih *Make a copy*.
- **Dari GitHub** — kalau folder ini diunggah ke sebuah repositori, Colab bisa
  membukanya lewat **File → Open notebook → GitHub**.
- **Jupyter di laptop** — juga bisa, asalkan sudah ada
  `pandas`, `scikit-learn`, `matplotlib`, dan `seaborn`.
  Tidak ada kode khusus Colab di dalam notebook-nya.

---

## Sumber data (disclosure)

Panduan LKS 2026 mewajibkan seluruh sumber data dicantumkan. Tabel ini bisa
langsung disalin ke proposal, dengan penyesuaian pada data yang benar-benar
dipakai.

| Notebook | Dataset | Sumber | Lisensi | Cara memuat |
|---|---|---|---|---|
| 01 | CSV kecil buatan sendiri | dibuat di dalam notebook | — | dibuat oleh selnya sendiri |
| 02, 03 | Palmer Penguins (344 baris) | Gorman KB, Williams TD, Fraser WR (2014), *PLoS ONE* 9(3):e90081; paket `palmerpenguins` (Horst, Hill & Gorman 2020) | CC0, domain publik | `pd.read_csv` dari `raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv` |
| 04 | California Housing (20.640 baris) | Sensus AS 1990 per *block group*; Pace RK & Barry R (1997), *Statistics & Probability Letters* 33(3):291–297; disediakan scikit-learn via StatLib | domain publik | `sklearn.datasets.fetch_california_housing()` |
| latihan 03, 04 | Konsumsi listrik sekolah (sintetis) | dibuat untuk paket materi ini, ada di `../03-praktikum/pengayaan-python/` | bebas dipakai | `pd.read_csv` berkas yang diunggah |

Kedua dataset utama berupa data terbuka dan tidak memuat data pribadi — sesuai
ketentuan **Batasan Data** pada panduan LKS 2026.

### Kalau Colab tidak bisa menjangkau internet

Notebook 02 dan 03 memuat data dari tautan. Tiga cadangan, berurutan:

1. `df = sns.load_dataset("penguins")` — seaborn punya salinan dataset yang sama.
2. Unduh CSV-nya lewat peramban, unggah ke Colab dengan
   `from google.colab import files; files.upload()`, lalu
   `pd.read_csv("penguins.csv")`.
3. Untuk notebook 04, `fetch_california_housing()` menyimpan hasil unduhannya,
   jadi cukup berhasil sekali. Jalankan sel itu lebih dulu saat masih ada
   internet.

**Sebelum hari-H, jalankan keempat notebook dari awal sampai akhir di jaringan
yang akan dipakai.** Ini pemeriksaan yang paling berguna dan paling sering
dilewatkan.

---

## Kaitan dengan penilaian LKS 2026

| Kriteria penilaian | Bobot | Bagian yang melatihnya |
|---|---|---|
| Pemahaman masalah & relevansi | 20% | delapan pertanyaan EDA (02 Bagian 4); penentuan target (03 Bagian 2) |
| Kreativitas & inovasi solusi | 20% | rekayasa fitur (04 Bagian 8) — satu gagasan mengalahkan semua penyetelan |
| Pemanfaatan AI yang tepat guna | 20% | baseline (03 Bagian 5); kapan `if` lebih baik daripada model (01 Bagian 4) |
| **Responsible AI** | **15%** | tabel risiko (03 Bagian 13, 04 Bagian 12); ambang keyakinan + penerusan ke manusia (03 Bagian 12); metrik per kelompok (04 Bagian 10) |
| Fungsionalitas prototipe | 15% | simpan model `.joblib` dan pakai pada data baru (03 Bagian 12, 04 Bagian 11) |
| Kejelasan presentasi | 10% | cara melaporkan hasil (03 Bagian 11, 04 Bagian 10) |

Materi ini melatih **bagian AI** dari aplikasi lomba. Pembungkusnya menjadi
aplikasi, proposal, dan video pitch ada pada berkas lain di paket ini:
`05-paket-simulasi-lks/` dan `03-praktikum/`.

---

## Catatan untuk pengajar

**Setiap notebook punya sel pemeriksaan otomatis** di bagian akhir, berisi
`assert` yang gagal dengan petunjuk bila jawabannya salah. Tidak disediakan
kunci jawaban terpisah — petunjuknya sudah ada di dalam selnya, dan semua yang
dibutuhkan ada di bagian sebelumnya. Kalau siswa mentok lebih dari 10 menit,
arahkan ke bagian yang bersangkutan, jangan berikan kodenya.

**Pembagian sesi yang wajar:**

| Sesi | Isi |
|---|---|
| 1 (3 jam) | notebook 01 |
| 2 (3 jam) | notebook 02 |
| 3 (3 jam) | notebook 03 sampai Bagian 8 |
| 4 (3 jam) | notebook 03 selesai + notebook 04 sampai Bagian 6 |
| 5 (3 jam) | notebook 04 selesai + latihan mandiri |

Kalau waktunya hanya dua hari: notebook 01 dipersingkat menjadi Bagian 1–7 saja,
dan notebook 04 dikerjakan dari Bagian 3 (bagian EDA-nya dijelaskan, tidak
dikerjakan sendiri).

**Empat hal yang paling sering menghambat di kelas:**

1. Sel dijalankan tidak berurutan → `NameError`. Ajarkan
   **Runtime → Run all** sejak awal.
2. Indentasi campur Tab dan spasi → `IndentationError`. Colab memakai 4 spasi.
3. Nama kolom salah ketik, termasuk beda huruf besar-kecil → `KeyError`.
   Ajarkan `df.columns` sebagai refleks.
4. Notebook 03 Bagian 10 dan notebook 04 Bagian 9 (`GridSearchCV`) memakan
   waktu 1–3 menit di Colab gratis. Beri tahu sebelumnya supaya tidak dianggap
   macet.

**Dua hal yang ditekankan berulang kali di seluruh materi**, karena keduanya yang
paling menentukan mutu pekerjaan dan paling sering ditanyakan juri:

- **Selalu punya baseline, dan selalu sebutkan jumlah datanya.**
  "69 dari 69 benar pada data uji" — bukan "akurasi 100%".
- **Data uji dibuka satu kali, di akhir.** Setiap penyimpangan dari ini
  menghasilkan angka yang tidak jujur.

---

## Berkas lain pada paket materi ini

```
00-README/                     Peta penggunaan seluruh paket. Baca ini dulu.
01-rencana-pembelajaran/       Jadwal, peran, asesmen, rencana cadangan.
02-modul-siswa/                Bacaan siswa, 5 bab + lembar kerja + glosarium.
03-praktikum/                  PilahPintar (klasifikasi gambar, tanpa kode) + pengayaan Python.
04-presentasi/                 Deck PPTX/PDF + catatan pemateri.
05-paket-simulasi-lks/         Problem canvas, disclosure, rubrik, panduan pitch, bank pertanyaan juri.
06-asesmen-dan-panduan-guru/   Pre/post-test, kunci, checklist, panduan fasilitator.
07-notebook-colab/             <- folder ini.
99-sumber/                     Klaim, sumber, dan status verifikasinya.
```

Hubungannya dengan jalur praktik utama: **PilahPintar** (di `03-praktikum/`)
mengajarkan klasifikasi gambar tanpa satu baris kode pun, memakai Teachable
Machine. Folder ini adalah jalur Python-nya — konsep yang sama (data latih
versus data uji, baseline, metrik, batasan model) dikerjakan sendiri dengan
kode, sehingga siswa bisa menjelaskan apa yang sebenarnya terjadi di dalam
Teachable Machine.

---

## Catatan verifikasi

Keempat notebook dijalankan dari awal sampai akhir sebelum diserahkan, dengan
`pandas` 3.0, `scikit-learn` 1.9, dan `seaborn` 0.13 — semuanya berjalan tanpa
galat. Kode ditulis agar juga berjalan pada versi yang dipakai Colab saat ini
(`pandas` 2.x, `scikit-learn` 1.6), dengan menghindari argumen yang berbeda
antar versi.

Yang **tidak** diverifikasi: kecepatan di Colab gratis pada jam sibuk, dan
ketersediaan tautan dataset dari jaringan sekolah. Keduanya perlu dicek sendiri
sebelum hari-H.

Angka hasil yang muncul pada teks penjelasan (akurasi, RMSE, MAE, jumlah baris,
nilai korelasi) diambil dari keluaran sungguhan, bukan perkiraan. Pada mesin
dengan versi pustaka berbeda, angkanya bisa bergeser sedikit — yang tidak
bergeser adalah kesimpulannya.
