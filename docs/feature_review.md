| Feature | Type | Decision | Reason |
|---------|------|----------|--------|
| srcip | nominal | DROP | Identifier (IP address, not useful for modeling) |
| sport | integer | KEEP | Port numbers are discrete, help distinguish service types (e.g., HTTP/SSH) |
| dstip | nominal | DROP | Identifier (IP address, not useful for modeling) |
| dsport | integer | KEEP | Port numbers are discrete, help distinguish service types (e.g., HTTP/SSH) |
| proto | nominal | KEEP | Potentially informative |
| state | nominal | KEEP | Connection state provides useful categorical info |
| dur | float | KEEP | Potentially informative |
| sbytes | integer | KEEP | Potentially informative |
| dbytes | integer | KEEP | Potentially informative |
| sttl | integer | KEEP | Potentially informative |
| dttl | integer | KEEP | Potentially informative |
| sloss | integer | KEEP | Potentially informative |
| dloss | integer | KEEP | Potentially informative |
| service | nominal | KEEP | Potentially informative |
| Sload | float | KEEP | Potentially informative |
| Dload | float | KEEP | Potentially informative |
| Spkts | integer | KEEP | Potentially informative |
| Dpkts | integer | KEEP | Potentially informative |
| swin | integer | KEEP | Potentially informative |
| dwin | integer | KEEP | Potentially informative |
| stcpb | integer | KEEP | Potentially informative |
| dtcpb | integer | KEEP | Potentially informative |
| smeansz | integer | KEEP | Potentially informative |
| dmeansz | integer | KEEP | Potentially informative |
| trans_depth | integer | KEEP | Potentially informative |
| res_bdy_len | integer | KEEP | Potentially informative |
| Sjit | float | KEEP | Potentially informative |
| Djit | float | KEEP | Potentially informative |
| Stime | timestamp | DROP | Timestamp (not used in current modeling task) |
| Ltime | timestamp | DROP | Timestamp (not used in current modeling task) |
| Sintpkt | float | KEEP | Potentially informative |
| Dintpkt | float | KEEP | Potentially informative |
| tcprtt | float | KEEP | Potentially informative |
| synack | float | KEEP | Potentially informative |
| ackdat | float | KEEP | Potentially informative |
| is_sm_ips_ports | binary | DROP | Identifier (IP address, not useful for modeling) |
| ct_state_ttl | integer | KEEP | Potentially informative |
| ct_flw_http_mthd | integer | DROP | Protocol-specific or sparse field (FTP/HTTP-related) |
| is_ftp_login | binary | DROP | Protocol-specific or sparse field (FTP/HTTP-related) |
| ct_ftp_cmd | integer | DROP | Protocol-specific or sparse field (FTP/HTTP-related) |
| ct_srv_src | integer | KEEP | Potentially informative |
| ct_srv_dst | integer | KEEP | Potentially informative |
| ct_dst_ltm | integer | KEEP | Potentially informative |
| ct_src_ ltm | integer | KEEP | Potentially informative |
| ct_src_dport_ltm | integer | KEEP | Potentially informative |
| ct_dst_sport_ltm | integer | KEEP | Potentially informative |
| ct_dst_src_ltm | integer | KEEP | Potentially informative |
| attack_cat | nominal | DROP | Supervised label or category, should not be used as input |
| Label | binary | DROP | Supervised label or category, should not be used as input |