class Transaction:

    FIXED_SIZE_BYTES = 256

    def __init__(self, tx_id, timestamp):
        self.tx_id = tx_id
        self.size_bytes = self.FIXED_SIZE_BYTES
        self.size_kb = self.size_bytes / 1024.0
        self.timestamp = timestamp