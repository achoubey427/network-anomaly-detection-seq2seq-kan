# Dataset Verification & Provenance Report

**Audit Date:** 2026-10-07  
**Protocol:** Autonomous Scientific Evaluation Protocol  
**Integrity Standard:** Zero Synthetic Data, Zero Fabricated Metrics, Strict Authoritative Origin Verification

---

## 1. Physical Disk Audit Summary

A systematic, non-destructive audit across local drives (`C:`, `D:`, `E:`) and `C:\Users\ASUS\Downloads\` was conducted to verify raw dataset availability.

| Benchmark Dataset | Target File / Archive | Expected Location | Disk Audit Status | Action Taken |
| :--- | :--- | :--- | :---: | :--- |
| **CSE-CIC-IDS2018** | `Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv` | `data/raw/CSECICIDS2018/` | **VERIFIED PRESENT** (102.85 MB, 331,125 rows) | **FROZEN & FULLY EVALUATED** |
| **CIC-IDS2017** | `MachineLearningCSV.zip` / `*.pcap_ISCX.csv` | `C:\Users\ASUS\Downloads\` or `data/raw/CICIDS2017/` | **ABSENT LOCALLY** | Awaiting genuine manual transfer / official file provision. |
| **CIC-DDoS2019** | `CSV-01-12.zip` / `CSV-03-11.zip` | `C:\Users\ASUS\Downloads\` or `data/raw/CICIDS2019/` | **ABSENT LOCALLY** | Partial `.crdownload` duplicate safely stopped and removed. |
| **CICIoT2023** | `*.csv` (46 features) | `data/raw/CICIDS2023/` or `data/raw/CICIoT2023/` | **BLOCKED (UNB PORTAL)** | Access restricted behind registration barrier; documented per protocol. |

---

## 2. Ingestion & Pipeline Readiness

The preprocessing engine, canonical schema normalizer, sequence builder, and experiment runners are completely prepared and verified:

1. **Adapters Active:**
   - `src/data_adapters/csecicids2018_adapter.py`
   - `src/data_adapters/cicids2017_adapter.py`
   - `src/data_adapters/cicids2019_adapter.py`
   - `src/data_adapters/cicids2023_adapter.py`
2. **Leakage Protection:**
   - Verified zero temporal leakage across all chronological splits.
   - Robust scaling parameters ($Q_{50}, IQR$) fitted strictly on the normal training partition.
3. **Regression Suite:**
   - **51 / 51 tests PASS** in `tests/`.

---

## 3. Immediate Next Step for Execution

Once the genuine files (`MachineLearningCSV.zip` or extracted CSVs for CIC-IDS2017, and `CSV-01-12.zip` / `CSV-03-11.zip` for CIC-DDoS2019) are placed into `C:\Users\ASUS\Downloads\` or directly into `data/raw/CICIDS2017/` / `data/raw/CICIDS2019/`, the pipeline will immediately:
1. Run schema profiling via `scripts/inspect_datasets.py`.
2. Generate temporal sequence tensors via `scripts/prepare_dataset.py --mode research`.
3. Sequentially train and evaluate Baseline Seq2Seq and Proposed KAN Seq2Seq models.
4. Generate publication-ready comparison figures and audit reports.
