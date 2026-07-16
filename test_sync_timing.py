from transcribe import SYNC_OUTPUT_LATENCY_SEC, calculate_sync_time_offset


def main():
    """驗證絕對網格、相對網格與記譜模式的共用時間偏移。"""
    assert calculate_sync_time_offset(20.0, True, True) == -SYNC_OUTPUT_LATENCY_SEC
    assert calculate_sync_time_offset(20.0, True, False) == 20.0 - SYNC_OUTPUT_LATENCY_SEC
    assert calculate_sync_time_offset(20.0, False, True) == 0.0
    print("sync timing self-check: PASS")


if __name__ == "__main__":
    main()
