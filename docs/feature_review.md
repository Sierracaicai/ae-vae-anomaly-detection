| Feature | Type | Decision | Reason |
|---------|------|----------|--------|
| srcip | nominal | DROP | Identifier (IP address, not useful for modeling) |
| sport | integer | KEEP | Useful session-level metadata — keep for modeling |
| dstip | nominal | DROP | Identifier (IP address, not useful for modeling) |
| dsport | integer | KEEP | Useful session-level metadata — keep for modeling |
| proto | nominal | KEEP | Network protocol type — differentiates TCP, UDP, ICMP |
| state | nominal | KEEP | Connection state — describes session outcome |
| dur | float | KEEP | Duration of connection — indicates session length |
| sbytes | integer | KEEP | Data volume — helps detect abnormal payloads |
| dbytes | integer | KEEP | Data volume — helps detect abnormal payloads |
| sttl | integer | KEEP | Time-to-live — reflects routing behavior |
| dttl | integer | KEEP | Time-to-live — reflects routing behavior |
| sloss | integer | KEEP | Useful session-level metadata — keep for modeling |
| dloss | integer | KEEP | Useful session-level metadata — keep for modeling |
| service | nominal | KEEP | Useful session-level metadata — keep for modeling |
| Sload | float | KEEP | Traffic rate — can show spikes or slowdowns |
| Dload | float | KEEP | Traffic rate — can show spikes or slowdowns |
| Spkts | integer | KEEP | Packet count — traffic volume per session |
| Dpkts | integer | KEEP | Packet count — traffic volume per session |
| swin | integer | KEEP | TCP window size — indicative of flow control behavior |
| dwin | integer | KEEP | TCP window size — indicative of flow control behavior |
| stcpb | integer | KEEP | TCP base sequence number — may reflect anomalies |
| dtcpb | integer | KEEP | TCP base sequence number — may reflect anomalies |
| smeansz | integer | KEEP | Mean packet size — useful for flow profiling |
| dmeansz | integer | KEEP | Mean packet size — useful for flow profiling |
| trans_depth | integer | KEEP | HTTP transaction depth — indicates request complexity |
| res_bdy_len | integer | KEEP | Response body size — proxy for data served |
| Sjit | float | KEEP | Jitter — packet timing variation may indicate instability |
| Djit | float | KEEP | Jitter — packet timing variation may indicate instability |
| Stime | timestamp | DROP | Timestamp (not used in current modeling task) |
| Ltime | timestamp | DROP | Timestamp (not used in current modeling task) |
| Sintpkt | float | KEEP | Inter-packet interval — traffic burstiness measure |
| Dintpkt | float | KEEP | Inter-packet interval — traffic burstiness measure |
| tcprtt | float | KEEP | TCP timing — handshake and round-trip latency |
| synack | float | KEEP | TCP timing — handshake and round-trip latency |
| ackdat | float | KEEP | TCP timing — handshake and round-trip latency |
| is_sm_ips_ports | binary | DROP | Identifier (IP address, not useful for modeling) |
| ct_state_ttl | integer | KEEP | Time-to-live — reflects routing behavior |
| ct_flw_http_mthd | integer | DROP | Protocol-specific or sparse field (FTP/HTTP-related) |
| is_ftp_login | binary | DROP | Protocol-specific or sparse field (FTP/HTTP-related) |
| ct_ftp_cmd | integer | DROP | Protocol-specific or sparse field (FTP/HTTP-related) |
| ct_srv_src | integer | KEEP | Aggregated service-based feature (e.g., per host stats) |
| ct_srv_dst | integer | KEEP | Aggregated service-based feature (e.g., per host stats) |
| ct_dst_ltm | integer | KEEP | Connection history in last 2s — recent activity stats |
| ct_src_ ltm | integer | KEEP | Connection history in last 2s — recent activity stats |
| ct_src_dport_ltm | integer | KEEP | Connection history in last 2s — recent activity stats |
| ct_dst_sport_ltm | integer | KEEP | Connection history in last 2s — recent activity stats |
| ct_dst_src_ltm | integer | KEEP | Connection history in last 2s — recent activity stats |
| attack_cat | nominal | DROP | Supervised label or category, should not be used as input |
| Label | binary | DROP | Supervised label or category, should not be used as input |