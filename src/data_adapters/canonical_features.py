"""
Canonical Feature Alignment and Non-Learning Column Filtering.
Provides explicit mappings between raw dataset column names and canonical network-flow features,
as well as identifying non-learning ID and leakage columns to remove.
"""

from typing import Dict, List, Set, Tuple


# Standard canonical network flow features
CANONICAL_FLOW_FEATURES = [
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "fwd_packet_length_max",
    "fwd_packet_length_min",
    "fwd_packet_length_mean",
    "fwd_packet_length_std",
    "bwd_packet_length_max",
    "bwd_packet_length_min",
    "bwd_packet_length_mean",
    "bwd_packet_length_std",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_max",
    "flow_iat_min",
    "fwd_iat_tot",
    "fwd_iat_mean",
    "fwd_iat_std",
    "fwd_iat_max",
    "fwd_iat_min",
    "bwd_iat_tot",
    "bwd_iat_mean",
    "bwd_iat_std",
    "bwd_iat_max",
    "bwd_iat_min",
    "fwd_header_length",
    "bwd_header_length",
    "fwd_packets_per_sec",
    "bwd_packets_per_sec",
    "packet_length_min",
    "packet_length_max",
    "packet_length_mean",
    "packet_length_std",
    "packet_length_variance",
    "fin_flag_count",
    "syn_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "ack_flag_count",
    "urg_flag_count",
    "down_up_ratio",
    "avg_packet_size",
    "init_fwd_win_bytes",
    "init_bwd_win_bytes",
]

# Non-learning / ID / Potential Leakage Columns to discard from feature matrices
DISCARD_COLUMNS: Set[str] = {
    "flow id",
    "flow_id",
    "id",
    "source ip",
    "src ip",
    "src_ip",
    "destination ip",
    "dst ip",
    "dst_ip",
    "timestamp",
    "time",
    "simillarhttp",
    "similarhttp",
    "unnamed: 0",
    "row_id",
    "filename",
    "source_file",
    "label",
    "class",
    "attack",
    "binary_label",
    "raw_label",
    "is_anomaly",
    "is_attack"
}

# Raw column mapping to canonical names per dataset
DATASET_FEATURE_MAPPINGS: Dict[str, Dict[str, str]] = {
    "CICIDS2017": {
        "flow duration": "flow_duration",
        "total fwd packets": "total_fwd_packets",
        "total backward packets": "total_bwd_packets",
        "total length of fwd packets": "total_fwd_bytes",
        "total length of bwd packets": "total_bwd_bytes",
        "fwd packet length max": "fwd_packet_length_max",
        "fwd packet length min": "fwd_packet_length_min",
        "fwd packet length mean": "fwd_packet_length_mean",
        "fwd packet length std": "fwd_packet_length_std",
        "bwd packet length max": "bwd_packet_length_max",
        "bwd packet length min": "bwd_packet_length_min",
        "bwd packet length mean": "bwd_packet_length_mean",
        "bwd packet length std": "bwd_packet_length_std",
        "flow bytes/s": "flow_bytes_per_sec",
        "flow packets/s": "flow_packets_per_sec",
        "flow iat mean": "flow_iat_mean",
        "flow iat std": "flow_iat_std",
        "flow iat max": "flow_iat_max",
        "flow iat min": "flow_iat_min",
        "fwd iat total": "fwd_iat_tot",
        "fwd iat mean": "fwd_iat_mean",
        "fwd iat std": "fwd_iat_std",
        "fwd iat max": "fwd_iat_max",
        "fwd iat min": "fwd_iat_min",
        "bwd iat total": "bwd_iat_tot",
        "bwd iat mean": "bwd_iat_mean",
        "bwd iat std": "bwd_iat_std",
        "bwd iat max": "bwd_iat_max",
        "bwd iat min": "bwd_iat_min",
        "fwd header length": "fwd_header_length",
        "bwd header length": "bwd_header_length",
        "fwd packets/s": "fwd_packets_per_sec",
        "bwd packets/s": "bwd_packets_per_sec",
        "min packet length": "packet_length_min",
        "max packet length": "packet_length_max",
        "packet length mean": "packet_length_mean",
        "packet length std": "packet_length_std",
        "packet length variance": "packet_length_variance",
        "fin flag count": "fin_flag_count",
        "syn flag count": "syn_flag_count",
        "rst flag count": "rst_flag_count",
        "psh flag count": "psh_flag_count",
        "ack flag count": "ack_flag_count",
        "urg flag count": "urg_flag_count",
        "down/up ratio": "down_up_ratio",
        "average packet size": "avg_packet_size",
        "init_win_bytes_forward": "init_fwd_win_bytes",
        "init_win_bytes_backward": "init_bwd_win_bytes",
    },
    "CICIDS2019": {
        "flow duration": "flow_duration",
        "tot fwd pkts": "total_fwd_packets",
        "tot bwd pkts": "total_bwd_packets",
        "totlen fwd pkts": "total_fwd_bytes",
        "totlen bwd pkts": "total_bwd_bytes",
        "fwd pkt len max": "fwd_packet_length_max",
        "fwd pkt len min": "fwd_packet_length_min",
        "fwd pkt len mean": "fwd_packet_length_mean",
        "fwd pkt len std": "fwd_packet_length_std",
        "bwd pkt len max": "bwd_packet_length_max",
        "bwd pkt len min": "bwd_packet_length_min",
        "bwd pkt len mean": "bwd_packet_length_mean",
        "bwd pkt len std": "bwd_packet_length_std",
        "flow byts/s": "flow_bytes_per_sec",
        "flow pkts/s": "flow_packets_per_sec",
        "flow iat mean": "flow_iat_mean",
        "flow iat std": "flow_iat_std",
        "flow iat max": "flow_iat_max",
        "flow iat min": "flow_iat_min",
        "fwd iat tot": "fwd_iat_tot",
        "fwd iat mean": "fwd_iat_mean",
        "fwd iat std": "fwd_iat_std",
        "fwd iat max": "fwd_iat_max",
        "fwd iat min": "fwd_iat_min",
        "bwd iat tot": "bwd_iat_tot",
        "bwd iat mean": "bwd_iat_mean",
        "bwd iat std": "bwd_iat_std",
        "bwd iat max": "bwd_iat_max",
        "bwd iat min": "bwd_iat_min",
        "fwd header len": "fwd_header_length",
        "bwd header len": "bwd_header_length",
        "fwd pkts/s": "fwd_packets_per_sec",
        "bwd pkts/s": "bwd_packets_per_sec",
        "pkt len min": "packet_length_min",
        "pkt len max": "packet_length_max",
        "pkt len mean": "packet_length_mean",
        "pkt len std": "packet_length_std",
        "pkt len var": "packet_length_variance",
        "fin flag cnt": "fin_flag_count",
        "syn flag cnt": "syn_flag_count",
        "rst flag cnt": "rst_flag_count",
        "psh flag cnt": "psh_flag_count",
        "ack flag cnt": "ack_flag_count",
        "urg flag cnt": "urg_flag_count",
        "down/up ratio": "down_up_ratio",
        "pkt size avg": "avg_packet_size",
        "init fwd win byts": "init_fwd_win_bytes",
        "init bwd win byts": "init_bwd_win_bytes",
    },
    "CICIDS2023": {
        "flow_duration": "flow_duration",
        "header_length": "fwd_header_length",
        "duration": "flow_duration",
        "rate": "flow_packets_per_sec",
        "srate": "fwd_packets_per_sec",
        "drate": "bwd_packets_per_sec",
        "min": "packet_length_min",
        "max": "packet_length_max",
        "avg": "packet_length_mean",
        "std": "packet_length_std",
        "tot sum": "total_fwd_bytes",
        "tot size": "avg_packet_size",
        "iat": "flow_iat_mean",
        "magnitude": "packet_length_mean",
        "variance": "packet_length_variance",
        "weight": "total_fwd_packets",
    },
    "CSECICIDS2018": {
        "dst port": "dst_port",
        "protocol": "protocol",
        "flow duration": "flow_duration",
        "tot fwd pkts": "total_fwd_packets",
        "tot bwd pkts": "total_bwd_packets",
        "totlen fwd pkts": "total_fwd_bytes",
        "totlen bwd pkts": "total_bwd_bytes",
        "fwd pkt len max": "fwd_packet_length_max",
        "fwd pkt len min": "fwd_packet_length_min",
        "fwd pkt len mean": "fwd_packet_length_mean",
        "fwd pkt len std": "fwd_packet_length_std",
        "bwd pkt len max": "bwd_packet_length_max",
        "bwd pkt len min": "bwd_packet_length_min",
        "bwd pkt len mean": "bwd_packet_length_mean",
        "bwd pkt len std": "bwd_packet_length_std",
        "flow byts/s": "flow_bytes_per_sec",
        "flow pkts/s": "flow_packets_per_sec",
        "flow iat mean": "flow_iat_mean",
        "flow iat std": "flow_iat_std",
        "flow iat max": "flow_iat_max",
        "flow iat min": "flow_iat_min",
        "fwd iat tot": "fwd_iat_tot",
        "fwd iat mean": "fwd_iat_mean",
        "fwd iat std": "fwd_iat_std",
        "fwd iat max": "fwd_iat_max",
        "fwd iat min": "fwd_iat_min",
        "bwd iat tot": "bwd_iat_tot",
        "bwd iat mean": "bwd_iat_mean",
        "bwd iat std": "bwd_iat_std",
        "bwd iat max": "bwd_iat_max",
        "bwd iat min": "bwd_iat_min",
        "fwd header len": "fwd_header_length",
        "bwd header len": "bwd_header_length",
        "fwd pkts/s": "fwd_packets_per_sec",
        "bwd pkts/s": "bwd_packets_per_sec",
        "pkt len min": "packet_length_min",
        "pkt len max": "packet_length_max",
        "pkt len mean": "packet_length_mean",
        "pkt len std": "packet_length_std",
        "pkt len var": "packet_length_variance",
        "fin flag cnt": "fin_flag_count",
        "syn flag cnt": "syn_flag_count",
        "rst flag cnt": "rst_flag_count",
        "psh flag cnt": "psh_flag_count",
        "ack flag cnt": "ack_flag_count",
        "urg flag cnt": "urg_flag_count",
        "down/up ratio": "down_up_ratio",
        "pkt size avg": "avg_packet_size",
        "init fwd win byts": "init_fwd_win_bytes",
        "init bwd win byts": "init_bwd_win_bytes",
    },
    "CSE-CIC-IDS2018": {
        "flow duration": "flow_duration",
        "tot fwd pkts": "total_fwd_packets",
        "tot bwd pkts": "total_bwd_packets",
        "totlen fwd pkts": "total_fwd_bytes",
        "totlen bwd pkts": "total_bwd_bytes",
        "flow byts/s": "flow_bytes_per_sec",
        "flow pkts/s": "flow_packets_per_sec",
    }
}


def sanitize_column_name(col: str) -> str:
    """Normalize raw column name: strip whitespace and convert to lowercase."""
    return col.strip().lower()


def map_raw_columns(raw_columns: List[str], dataset_id: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Given raw column names, returns:
    1. retained_raw_cols: raw column names to keep for features
    2. canonical_col_names: corresponding canonical names
    3. discarded_cols: raw column names dropped as IDs, timestamps, or leakage
    """
    mapping = DATASET_FEATURE_MAPPINGS.get(dataset_id, {})
    retained_raw = []
    canonical_names = []
    discarded = []

    for raw_col in raw_columns:
        sanitized = sanitize_column_name(raw_col)
        
        # Check if column is an ID/leakage column
        if sanitized in DISCARD_COLUMNS:
            discarded.append(raw_col)
            continue
            
        # Check if mapped to canonical feature
        if sanitized in mapping:
            canonical_name = mapping[sanitized]
            retained_raw.append(raw_col)
            canonical_names.append(canonical_name)
        else:
            # Fallback: retain sanitized feature name if not an ID/label
            retained_raw.append(raw_col)
            canonical_names.append(sanitized.replace(" ", "_").replace("/", "_per_"))

    return retained_raw, canonical_names, discarded
