# Comparative Analysis of Benchmark Intrusion Detection Datasets

This document provides a scientific analysis and comparative schema breakdown of the four network intrusion datasets targeted in the 4 &times; 2 research matrix.

---

## 1. Overview Matrix

| Dimension | CICIDS2017 | CICIDS2019 *(CSE-CIC-IDS2018 / CICDDoS2019)* | CICIoT2023 | CICIDS2025 *(CIC-DGG-2025)* |
|---|---|---|---|---|
| **Official Name** | ISCX CICIDS 2017 | CSE-CIC-IDS2018 / CIC-DDoS2019 | CICIoT 2023 | CIC-DGG-2025 / Faculty Target |
| **Originating Body** | Canadian Institute for Cybersecurity (UNB) | Communications Security Establishment & CIC | Canadian Institute for Cybersecurity | Faculty / CIC Modern Threat Track |
| **Primary Domain** | Enterprise Perimeter Networks | High-volume Modern DDoS & Infrastructure | IoT Topologies (Smart Home, Sensors) | Distributed Generative Cyber-threats |
| **Data Format** | CSV (CICFlowMeter output) | CSV (Standard / Whitespace headers) | CSV (Standardized flow attributes) | CSV / Flow Telemetry |
| **Raw Column Count** | 79 &ndash; 85 features | 80 &ndash; 88 features | 47 features | Dynamic (Variable schema) |
| **Target Label Column** | `Label` | `Label` | `label` | `auto_detect` (`Label` / `class`) |
| **Normal Class Identifier** | `BENIGN` | `BENIGN` | `BenignTraffic` | `BENIGN` / `Normal` |
| **Anomaly Classes** | DoS, DDoS, PortScan, Infiltration, Brute Force, Web Attacks | NetBIOS, LDAP, MSSQL, Syn, UDP, PortScan | 33 attack classes (Mirai, Recon, Flood, Spoofing) | Zero-day evasion, distributed telemetry |
| **Temporal Alignment** | Timestamp strings available | Timestamp strings available | Inter-arrival times & Flow Durations | Flow temporal sequencing |

---

## 2. In-Depth Dataset Profiles

### A. ISCX CICIDS 2017
- **Characteristics:** Captured over 5 business days (Monday to Friday) under realistic background traffic profiles and controlled attack injections.
- **Header Structure:** Column names contain whitespace and occasional trailing spaces (e.g. ` Destination Port`, ` Flow Duration`). Handled automatically via `sanitize_column_name`.
- **Anomalies:** Monday is strictly benign. Tuesday through Friday contain distinct attack scenarios (Brute Force SSH/FTP, DoS Slowloris, Heartbleed, Web Attacks, Infiltration, Botnet, DDoS).
- **Preprocessing Strategy:** All non-learning columns (`Flow ID`, `Source IP`, `Destination IP`, `Timestamp`) are filtered out to prevent shortcut learning and identity memorization.

### B. CICIDS 2019 *(CSE-CIC-IDS2018 / CIC-DDoS2019)*
- **Characteristics:** Focused on modern high-volume reflection and amplification attacks (NTP, DNS, LDAP, MSSQL, NetBIOS, SNMP, SSDP, UDP).
- **Header Structure:** Contains condensed feature names (`Tot Fwd Pkts`, `Totlen Bwd Pkts`, `Flow Byts/s`, `Init Fwd Win Byts`).
- **Feature Alignment:** Mapped to standard canonical names in `src/data_adapters/canonical_features.py`.

### C. CICIoT 2023
- **Characteristics:** Designed for real-world Internet of Things (IoT) network topologies containing 105 IoT devices across 33 attack types and benign traffic.
- **Header Structure:** Lowercase headers (`flow_duration`, `header_length`, `rate`, `srate`, `drate`, `magnitude`, `variance`).
- **Feature Alignment:** Mapped directly to canonical network rate and length statistics.

### D. CICIDS 2025 *(CIC-DGG-2025 / Faculty Target Dataset)*
- **Critical Policy:** The system enforces strict empirical verification. If the local faculty files represent `CIC-DGG-2025`, the dynamic adapter inspects column headers, infers numerical vs. categorical channels, and validates compatibility without making false assumptions.
- **Status:** Handled via `CICIDS2025Adapter` with dynamic schema detection and transparent reporting.

---

## 3. Canonical Feature Mapping Architecture

```
Raw Column (CICIDS2017)        Raw Column (CICIDS2019)        Raw Column (CICIoT2023)
      │                              │                               │
      ▼                              ▼                               ▼
" Total Fwd Packets"           "Tot Fwd Pkts"                "Weight" / "Tot sum"
      │                              │                               │
      └──────────────────────────────┼───────────────────────────────┘
                                     ▼
                      ┌────────────────────────────┐
                      │    CANONICAL FEATURE       │
                      │   `total_fwd_packets`      │
                      └────────────────────────────┘
```

---

## 4. Leakage Protection & Partitioning Rules

1. **Feature Exclusion:** Target labels (`raw_label`, `binary_label`, `Label`, `class`) are detached immediately and never enter feature tensor $X$.
2. **Scaler Isolation:** `RobustScaler` is fitted **strictly on normal training samples ($y=0$)**. Validation and test partitions are transformed using the saved training statistics.
3. **Partition Independence:** Sequence indices for Train (70%), Validation (15%), and Test (15%) are verified to have zero intersection.
4. **Frozen Validation Threshold:** Optimal decision threshold $\theta^*$ is determined on the validation partition and frozen before test set evaluation.

---

## 5. Dataset Incompatibilities & Limitations

- **Protocol Differences:** CICIDS2017/2019 flows contain TCP window flags that are not present in raw IoT packet streams (CICIoT2023). Unmapped columns are safely handled with median imputation or zero-padding.
- **Label Imbalance:** Attack proportions vary across benchmarks (ranging from 1% up to 80%). The sequence buffer ensures that window-level anomaly labels reflect actual presence ($\ge 1$ anomalous packet in window).
