import pickle
import numpy as np


def load_data(filename=r'E:\shiyan\CNN2\APC-CNNGRU-Attn-RML22\data\RML22.01A.pkl'):

    Xd = pickle.load(open(filename, 'rb'), encoding='iso-8859-1')


    mods = sorted(list(set([k[0] for k in Xd.keys()])))
    snrs = sorted(list(set([k[1] for k in Xd.keys()])))

    X = []
    lbl = []
    train_idx = []
    val_idx = []
    np.random.seed(2016)


    samples_per_group = Xd[(mods[0], snrs[0])].shape[0]
    total_groups = len(mods) * len(snrs)

    for group_idx, (mod, snr) in enumerate(Xd.keys()):

        group_data = Xd[(mod, snr)]
        assert group_data.shape[0] == samples_per_group, "检测到异常组样本量，需检查RML22数据结构"

        X.append(group_data)

        lbl.extend([(mod, snr)] * samples_per_group)


        group_start = group_idx * samples_per_group
        group_end = (group_idx + 1) * samples_per_group
        group_indices = list(range(group_start, group_end))


        train_size = int(samples_per_group * 0.6)
        val_size = int(samples_per_group * 0.2)
        test_size = samples_per_group - train_size - val_size


        train_selected = np.random.choice(group_indices, size=train_size, replace=False)
        train_idx.extend(train_selected)


        remaining = list(set(group_indices) - set(train_selected))
        val_selected = np.random.choice(remaining, size=val_size, replace=False)
        val_idx.extend(val_selected)


        test_selected = list(set(remaining) - set(val_selected))



    X = np.vstack(X)
    n_examples = X.shape[0]


    all_indices = set(range(n_examples))
    test_idx = list(all_indices - set(train_idx) - set(val_idx))


    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)


    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]


    def to_onehot(yy):
        yy1 = np.zeros([len(yy), len(mods)])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1


    Y_train = to_onehot([mods.index(lbl[i][0]) for i in train_idx])
    Y_val = to_onehot([mods.index(lbl[i][0]) for i in val_idx])
    Y_test = to_onehot([mods.index(lbl[i][0]) for i in test_idx])


    print(f"RML2022.10A数据集加载完成，总样本量: {n_examples}（预期462000）")
    print(f"训练集/验证集/测试集样本量: {len(X_train)}/{len(X_val)}/{len(X_test)}（预期277200/92400/92400）")
    print(f"标签形状（训练/验证/测试）: {Y_train.shape}/{Y_val.shape}/{Y_test.shape}")

    return (mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx)


if __name__ == '__main__':
    (mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (
    train_idx, val_idx, test_idx) = load_data()
