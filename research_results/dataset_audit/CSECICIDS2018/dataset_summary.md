# Dataset Audit Report: CSECICIDS2018
**Verification Status:** `READY`  
**Files Found:** 1  
**Total Raw Rows:** 331,125  
**Total Columns Detected:** 80  
**Retained Canonical Features:** 78  

## File Details

| Filename | Size (MB) | Rows |
|---|---|---|
| `Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv` | 102.85 | 331,125 |

## Data Integrity & Quality

- **Missing Values Detected:** 1,834
- **Infinite Values Detected:** 4,004
- **Unique Label Classes:** 3

## Class Distribution

| Class / Attack Label | Count | Percentage |
|---|---|---|
| `Benign` | 238,037 | 71.89% |
| `Label` | 25 | 0.01% |
| `Infilteration` | 93,063 | 28.11% |

## Feature Alignment & Discarded Columns

### Retained Canonical Features:
`dst_port`, `protocol`, `flow_duration`, `total_fwd_packets`, `total_bwd_packets`, `total_fwd_bytes`, `total_bwd_bytes`, `fwd_packet_length_max`, `fwd_packet_length_min`, `fwd_packet_length_mean`, `fwd_packet_length_std`, `bwd_packet_length_max`, `bwd_packet_length_min`, `bwd_packet_length_mean`, `bwd_packet_length_std`, `flow_bytes_per_sec`, `flow_packets_per_sec`, `flow_iat_mean`, `flow_iat_std`, `flow_iat_max`, `flow_iat_min`, `fwd_iat_tot`, `fwd_iat_mean`, `fwd_iat_std`, `fwd_iat_max`, `fwd_iat_min`, `bwd_iat_tot`, `bwd_iat_mean`, `bwd_iat_std`, `bwd_iat_max`, `bwd_iat_min`, `fwd_psh_flags`, `bwd_psh_flags`, `fwd_urg_flags`, `bwd_urg_flags`, `fwd_header_length`, `bwd_header_length`, `fwd_packets_per_sec`, `bwd_packets_per_sec`, `packet_length_min`, `packet_length_max`, `packet_length_mean`, `packet_length_std`, `packet_length_variance`, `fin_flag_count`, `syn_flag_count`, `rst_flag_count`, `psh_flag_count`, `ack_flag_count`, `urg_flag_count`, `cwe_flag_count`, `ece_flag_cnt`, `down_up_ratio`, `avg_packet_size`, `fwd_seg_size_avg`, `bwd_seg_size_avg`, `fwd_byts_per_b_avg`, `fwd_pkts_per_b_avg`, `fwd_blk_rate_avg`, `bwd_byts_per_b_avg`, `bwd_pkts_per_b_avg`, `bwd_blk_rate_avg`, `subflow_fwd_pkts`, `subflow_fwd_byts`, `subflow_bwd_pkts`, `subflow_bwd_byts`, `init_fwd_win_bytes`, `init_bwd_win_bytes`, `fwd_act_data_pkts`, `fwd_seg_size_min`, `active_mean`, `active_std`, `active_max`, `active_min`, `idle_mean`, `idle_std`, `idle_max`, `idle_min`

### Discarded Non-Learning / ID / Leakage Columns:
`Timestamp`, `Label`
