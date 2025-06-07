import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# -------------------
# 1) Static threshold evaluation (của bạn)
# -------------------
def load_data():
    data = fetch_olivetti_faces(shuffle=True, random_state=0)
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

def extract_embeddings(X_train, X_test, n_components=50):
    pca = PCA(n_components=n_components, whiten=True, random_state=0)
    return pca.fit_transform(X_train), pca.transform(X_test)

def evaluate_thresholds(X_train, y_train, X_test, y_test, thresholds):
    D = euclidean_distances(X_test, X_train)
    stats = {k: [] for k in
             ('threshold','accuracy','precision','recall','f1','far','frr')}
    for thr in thresholds:
        y_pred = []
        for dists, true in zip(D, y_test):
            idx = np.argmin(dists)
            y_pred.append(y_train[idx] if dists[idx] < thr else -1)
        # cơ bản metrics
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)
        # FAR / FRR
        false_accepts  = sum((p != t) and (p != -1)
                             for p,t in zip(y_pred, y_test))
        false_rejects  = sum((p == -1) and (t != -1)
                             for p,t in zip(y_pred, y_test))
        far = false_accepts  / len(y_test)
        frr = false_rejects  / len(y_test)
        # lưu
        for k,v in zip(stats,
                       (thr,acc,prec,rec,f1,far,frr)):
            stats[k].append(v)
    return stats

# -------------------
# 2) Định nghĩa LSPI để học policy ngưỡng
# -------------------
def generate_transitions(distances, y_true, thresholds):
    transitions = []
    for dist, true in zip(distances, y_true):
        s = np.array([dist])
        for a_idx, thr in enumerate(thresholds):
            pred = true if dist < thr else -1
            r = 1 if pred == true else -1
            transitions.append((s, a_idx, r, s))  # self-loop
    return transitions

def phi(s, a_idx, thresholds):
    onehot = np.zeros(len(thresholds))
    onehot[a_idx] = 1
    return np.concatenate((onehot, s))

def lspi(transitions, thresholds,
         gamma=0.9, max_iter=30, tol=1e-4, ridge=1e-5):
    nA = len(thresholds)
    d  = len(transitions[0][0]) + nA
    theta = np.zeros(d)
    # khởi policy: giữa dãy thresholds
    pi = np.full(len(transitions), nA//2, dtype=int)
    for _ in range(max_iter):
        A = np.eye(d)*ridge
        b = np.zeros(d)
        # Policy evaluation
        for idx, (s,a,r,s2) in enumerate(transitions):
            f  = phi(s, a, thresholds)
            f2 = phi(s2, pi[idx], thresholds)
            A += np.outer(f, f - gamma*f2)
            b += f * r
        theta_new = np.linalg.solve(A, b)
        if np.linalg.norm(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new
        # Policy improvement
        for idx, (s,_,_,_) in enumerate(transitions):
            q = [phi(s,a_idx,thresholds).dot(theta)
                 for a_idx in range(nA)]
            pi[idx] = np.argmax(q)
    return theta

def infer_threshold(theta, thresholds, dist):
    s = np.array([dist])
    q = [phi(s, i, thresholds).dot(theta)
         for i in range(len(thresholds))]
    return thresholds[np.argmax(q)]

def evaluate_lspi_policy(X_train, y_train, X_test, y_test, thresholds, theta):
    # tính distances và nearest index
    D   = euclidean_distances(X_test, X_train)
    idx = np.argmin(D, axis=1)
    dists = D.min(axis=1)
    # dự đoán theo policy learned
    y_pred = []
    for dist, i in zip(dists, idx):
        thr = infer_threshold(theta, thresholds, dist)
        y_pred.append(y_train[i] if dist < thr else -1)
    # tính các metric giống static
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)
    far = sum((p!=t) and (p!=-1) for p,t in zip(y_pred,y_test))/len(y_test)
    frr= sum((p==-1) and (t!=-1) for p,t in zip(y_pred,y_test))/len(y_test)
    return {'accuracy':acc,'precision':prec,'recall':rec,'f1':f1,'far':far,'frr':frr}

# -------------------
# 3) Main và Plot
# -------------------
def plot_comparison(stats_static, stats_lspi):
    plt.figure(figsize=(10,6))
    T = stats_static['threshold']
    plt.plot(T, stats_static['accuracy'], marker='o', label='Static Accuracy')
    plt.plot(T, stats_static['precision'], marker='s', label='Static Precision')
    plt.plot(T, stats_static['recall'], marker='^', label='Static Recall')
    plt.plot(T, stats_static['f1'], marker='d', label='Static F1-score')
    plt.plot(T, stats_static['far'], linestyle='--', label='Static FAR')
    plt.plot(T, stats_static['frr'], linestyle='--', label='Static FRR')
    # LSPI lines
    for m, v in stats_lspi.items():
        plt.hlines(v, T[0], T[-1],
                   linestyles='-.',
                   label=f'LSPI {m.capitalize()} = {v:.3f}')
    plt.title("Static vs. LSPI-Learned Threshold Policy Metrics")
    plt.xlabel("Threshold (Euclidean Distance)")
    plt.ylabel("Metric Value")
    plt.legend(bbox_to_anchor=(1.02,1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # load & embed
    X_train, X_test, y_train, y_test = load_data()
    X_train_emb, X_test_emb = extract_embeddings(X_train, X_test, n_components=50)

    # static thresholds
    thresholds = np.linspace(0.5, 6.0, 30)
    stats_static = evaluate_thresholds(
        X_train_emb, y_train, X_test_emb, y_test, thresholds
    )

    # LSPI
    dists       = euclidean_distances(X_test_emb, X_train_emb).min(axis=1)
    transitions = generate_transitions(dists, y_test, thresholds)
    theta       = lspi(transitions, thresholds, gamma=0.9, max_iter=30)
    stats_lspi  = evaluate_lspi_policy(
        X_train_emb, y_train, X_test_emb, y_test, thresholds, theta
    )

    # plot
    plot_comparison(stats_static, stats_lspi)

if __name__ == "__main__":
    main()
