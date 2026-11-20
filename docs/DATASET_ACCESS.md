# Authoritative Dataset Access & Ingestion Specification

**Document Version:** 1.0.0  
**Research Matrix:** 4 Intrusion Datasets &times; 2 Autoencoder Models (8 Controlled Experiments)

---

## 1. Dataset Matrix Overview

| Benchmark # | Dataset Name | Internal ID | Official UNB / CIC Source | Access Requirement | Raw Data Directory | Ingestion Readiness | Experiment Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **Benchmark 1** | **CIC-IDS2017** | `CICIDS2017` | [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html) | Academic User Agreement Form | `data/raw/CICIDS2017/` | **READY** *(Adapter Active)* | `PENDING RAW FILES` |
| **Benchmark 2** | **CIC-DDoS2019** | `CICIDS2019` | [https://www.unb.ca/cic/datasets/ddos-2019.html](https://www.unb.ca/cic/datasets/ddos-2019.html) | Academic User Agreement Form | `data/raw/CICIDS2019/` | **READY** *(Adapter Active)* | `PENDING RAW FILES` |
| **Benchmark 3** | **CICIoT2023** | `CICIDS2023` / `CICIoT2023` | [https://www.unb.ca/cic/datasets/iotdataset-2023.html](https://www.unb.ca/cic/datasets/iotdataset-2023.html) | Academic User Agreement Form | `data/raw/CICIDS2023/` | **READY** *(Adapter Active)* | `PENDING RAW FILES` |
| **Benchmark 4** | **CSE-CIC-IDS2018** | `CSECICIDS2018` | [AWS Open Data Registry](https://registry.opendata.aws/) (`s3://cse-cic-ids2018/`) | Public Anonymous Access | `data/raw/CSECICIDS2018/` | **ACQUIRED** *(102.85 MB)* | **COMPLETED & FROZEN** |

---

## 2. In-Depth Dataset Access & File Specifications

### A. Benchmark 1: ISCX CIC-IDS2017
* **Official Source:** Canadian Institute for Cybersecurity, University of New Brunswick (UNB)
* **Access URL:** `https://www.unb.ca/cic/datasets/ids-2017.html`
* **Access Mechanism:** Scroll to the bottom of the landing page, accept the research terms, and submit the academic registration form to receive download credentials.
* **Target Package:** `MachineLearningCSV.zip` (Processed CSV Flow Files).
* **Expected Files in `data/raw/CICIDS2017/`:**
  - `Monday-WorkingHours.pcap_ISCX.csv` (Benign baseline)
  - `Tuesday-WorkingHours.pcap_ISCX.csv` (FTP-Patator, SSH-Patator)
  - `Wednesday-workingHours.pcap_ISCX.csv` (DoS Slowloris, Slowhttptest, Hulk, GoldenEye, Heartbleed)
  - `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` (Web Attacks)
  - `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` (Infiltration)
  - `Friday-WorkingHours-Morning.pcap_ISCX.csv` (Botnet)
  - `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` (PortScan)
  - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` (DDoS)
* **Schema & Preprocessing:** 78 numerical features, `Label` column (`BENIGN` = 0, Attack = 1). Automatic sanitization of leading/trailing whitespaces.

---

### B. Benchmark 2: CIC-DDoS2019
* **Official Source:** Canadian Institute for Cybersecurity (UNB)
* **Access URL:** `https://www.unb.ca/cic/datasets/ddos-2019.html`
* **Access Mechanism:** Submit the UNB academic registration agreement form.
* **Target Package:** CSV flow datasets (Training Day / Testing Day CSVs).
* **Expected Files in `data/raw/CICIDS2019/`:**
  - `01-12/` or `03-11/` CSV files (e.g., `DrDoS_DNS.csv`, `DrDoS_LDAP.csv`, `DrDoS_MSSQL.csv`, `DrDoS_NetBIOS.csv`, `DrDoS_NTP.csv`, `DrDoS_SNMP.csv`, `DrDoS_SSDP.csv`, `DrDoS_UDP.csv`, `Syn.csv`, `UDPLag.csv`).
* **Schema & Preprocessing:** 80+ CICFlowMeter attributes with condensed abbreviations (`Tot Fwd Pkts`, `Flow Byts/s`). Label column: `Label`.

---

### C. Benchmark 3: CICIoT2023
* **Official Source:** Canadian Institute for Cybersecurity (UNB)
* **Access URL:** `https://www.unb.ca/cic/datasets/iotdataset-2023.html`
* **Access Mechanism:** Submit the UNB academic registration agreement form.
* **Target Package:** IoT Network CSV Flow Datasets.
* **Expected Files in `data/raw/CICIDS2023/` or `data/raw/CICIoT2023/`:**
  - Standardized split CSV files (e.g., `part-00000-...csv` or combined split CSVs).
* **Schema & Preprocessing:** 46 continuous topological flow features (`flow_duration`, `Header_Length`, `Rate`, `Srate`, `Drate`, `Magnitude`, `Variance`, etc.). Label column: `label` (`BenignTraffic` = 0, Attacks = 1).

---

### D. Benchmark 4: CSE-CIC-IDS2018 (FROZEN)
* **Official Source:** Communications Security Establishment (CSE) & CIC (UNB) / AWS Open Data
* **Status:** **ACQUIRED, EVALUATED & FROZEN**
* **Local File:** `data/raw/CSECICIDS2018/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv` (102.85 MB, 331,125 rows)
* **Experimental Metrics Saved:**
  - Baseline Seq2Seq: $F_1 = 0.9999$, $\text{Recall} = 1.0000$, $\text{Latency} = 23.00\text{ ms}$, $\text{ROC-AUC} = 0.5496$
  - KAN Autoencoder: $F_1 = 0.9999$, $\text{Recall} = 1.0000$, $\text{Latency} = 23.39\text{ ms}$, $\text{ROC-AUC} = 0.4994$

---

## 3. Autonomous Pipeline Execution Protocol

When raw CSV files are placed in their respective `data/raw/<DATASET_ID>/` folder, execute the automated pipeline with:

```bash
# 1. Profile and verify raw dataset schema
python scripts/inspect_datasets.py --dataset <DATASET_ID>

# 2. Preprocess and generate sequence tensors
python scripts/prepare_dataset.py --dataset <DATASET_ID> --mode research

# 3. Run Baseline Seq2Seq Experiment
python scripts/run_experiment.py --dataset <DATASET_ID> --model baseline_seq2seq

# 4. Run Proposed KAN Experiment
python scripts/run_experiment.py --dataset <DATASET_ID> --model kan_seq2seq

# 5. Generate Comparison and Publication Figures
python -c "from src.evaluation.comparator import ExperimentComparator; ExperimentComparator().generate_comparison_table()"
python scripts/generate_research_figures.py
```
