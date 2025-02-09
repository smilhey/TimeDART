


def train_val_test_split(data, patch_len, train_size=0.6, val_size=0.2, test_size=0.2):
    assert train_size + val_size + test_size == 1, "Sizes do not add up to 1"
    n = len(data)
    train_end = int(n * train_size)
    train_end = train_end - (train_end+1) % patch_len
    val_end = train_end + int(n * val_size)
    val_end = val_end - (val_end+1) % patch_len
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    return train, val, test
