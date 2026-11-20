# Dataset Audit Report: CICIDS2017
**Verification Status:** `READY`  
**Files Found:** 8  
**Total Raw Rows:** 2,830,743  
**Total Columns Detected:** 79  
**Retained Canonical Features:** 78  

## File Details

| Filename | Size (MB) | Rows |
|---|---|---|
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 73.55 | 225,745 |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 73.34 | 286,467 |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 55.62 | 191,033 |
| `Monday-WorkingHours.pcap_ISCX.csv` | 168.73 | 529,918 |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 79.25 | 288,602 |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 49.61 | 170,366 |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 128.82 | 445,909 |
| `Wednesday-workingHours.pcap_ISCX.csv` | 214.74 | 692,703 |

## Data Integrity & Quality

- **Missing Values Detected:** 1,358
- **Infinite Values Detected:** 4,376
- **Unique Label Classes:** 15

## Class Distribution

| Class / Attack Label | Count | Percentage |
|---|---|---|
| `BENIGN` | 2,273,097 | 80.3% |
| `DDoS` | 128,027 | 4.52% |
| `PortScan` | 158,930 | 5.61% |
| `Bot` | 1,966 | 0.07% |
| `Infiltration` | 36 | 0.0% |
| `Web Attack � Brute Force` | 1,507 | 0.05% |
| `Web Attack � XSS` | 652 | 0.02% |
| `Web Attack � Sql Injection` | 21 | 0.0% |
| `FTP-Patator` | 7,938 | 0.28% |
| `SSH-Patator` | 5,897 | 0.21% |
| `DoS slowloris` | 5,796 | 0.2% |
| `DoS Slowhttptest` | 5,499 | 0.19% |
| `DoS Hulk` | 231,073 | 8.16% |
| `DoS GoldenEye` | 10,293 | 0.36% |
| `Heartbleed` | 11 | 0.0% |

## Feature Alignment & Discarded Columns

### Retained Canonical Features:
`destination_port`, `flow_duration`, `total_fwd_packets`, `total_bwd_packets`, `total_fwd_bytes`, `total_bwd_bytes`, `fwd_packet_length_max`, `fwd_packet_length_min`, `fwd_packet_length_mean`, `fwd_packet_length_std`, `bwd_packet_length_max`, `bwd_packet_length_min`, `bwd_packet_length_mean`, `bwd_packet_length_std`, `flow_bytes_per_sec`, `flow_packets_per_sec`, `flow_iat_mean`, `flow_iat_std`, `flow_iat_max`, `flow_iat_min`, `fwd_iat_tot`, `fwd_iat_mean`, `fwd_iat_std`, `fwd_iat_max`, `fwd_iat_min`, `bwd_iat_tot`, `bwd_iat_mean`, `bwd_iat_std`, `bwd_iat_max`, `bwd_iat_min`, `fwd_psh_flags`, `bwd_psh_flags`, `fwd_urg_flags`, `bwd_urg_flags`, `fwd_header_length`, `bwd_header_length`, `fwd_packets_per_sec`, `bwd_packets_per_sec`, `packet_length_min`, `packet_length_max`, `packet_length_mean`, `packet_length_std`, `packet_length_variance`, `fin_flag_count`, `syn_flag_count`, `rst_flag_count`, `psh_flag_count`, `ack_flag_count`, `urg_flag_count`, `cwe_flag_count`, `ece_flag_count`, `down_up_ratio`, `avg_packet_size`, `avg_fwd_segment_size`, `avg_bwd_segment_size`, `fwd_header_length`, `fwd_avg_bytes_per_bulk`, `fwd_avg_packets_per_bulk`, `fwd_avg_bulk_rate`, `bwd_avg_bytes_per_bulk`, `bwd_avg_packets_per_bulk`, `bwd_avg_bulk_rate`, `subflow_fwd_packets`, `subflow_fwd_bytes`, `subflow_bwd_packets`, `subflow_bwd_bytes`, `init_fwd_win_bytes`, `init_bwd_win_bytes`, `act_data_pkt_fwd`, `min_seg_size_forward`, `active_mean`, `active_std`, `active_max`, `active_min`, `idle_mean`, `idle_std`, `idle_max`, `idle_min`

### Discarded Non-Learning / ID / Leakage Columns:
`Label`
