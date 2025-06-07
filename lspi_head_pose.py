import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
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
# 0) Loader + PCA embedding for your PersonDataset
# -------------------
def load_persondataset(base_dir, n_components=50):
    """
    base_dir/
      Person01-1/Personne01/*.jpg
      Person01-2/Personne01/*.jpg
      Person02-1/Personne02/*.jpg
      Person02-2/Personne02/*.jpg

    We flatten each gray image, assign an integer ID per PersonXX,
    then run PCA to n_components.  Returns:
      X_emb       : (N, D) float embeddings
      y_id        : (N,)   int identity labels
      pose_group  : (N,)   int 0 if folder *-1, 1 if *-2
    """
    images, labels, pose_group = [], [], []
    for person in sorted(os.listdir(base_dir)):
        person_dir = os.path.join(base_dir, person)
        if not os.path.isdir(person_dir):
            continue

        # parse "Person01-1" → id_num=0, grp_num=0 (for "-1")
        try:
            id_str, grp_str = person.split('-')
            id_num  = int(id_str.replace("Person", "")) - 1
            grp_num = int(grp_str) - 1
        except:
            continue

        # sometimes images live one level deeper:
        #  Person01-1/Personne01/*.jpg
        # so detect that
        contents = sorted(os.listdir(person_dir))
        # look for jpgs directly:
        jpgs = [f for f in contents if f.lower().endswith((".jpg", ".png"))]
        if jpgs:
            image_folder = person_dir
        else:
            # fallback: first subdir
            subs = [d for d in contents if os.path.isdir(os.path.join(person_dir, d))]
            if not subs:
                continue
            image_folder = os.path.join(person_dir, subs[0])

        # now load every image in image_folder
        for fname in sorted(os.listdir(image_folder)):
            if not fname.lower().endswith((".jpg", ".png")):
                continue
            path = os.path.join(image_folder, fname)
            img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            images.append(img.flatten())
            labels.append(id_num)
            pose_group.append(grp_num)

    if not images:
        raise RuntimeError(f"No images found under `{base_dir}`")

    X       = np.vstack(images)
    y_id    = np.array(labels,      dtype=int)
    pose_gp = np.array(pose_group,  dtype=int)

    pca   = PCA(n_components=n_components, whiten=True, random_state=0)
    X_emb = pca.fit_transform(X)
    return X_emb, y_id, pose_gp


# -------------------
# 1) Static threshold evaluation
# -------------------
def evaluate_thresholds(X_tr, y_tr, X_te, y_te, thresholds):
    D = euclidean_distances(X_te, X_tr)
    stats = {k: [] for k in
             ('threshold','accuracy','precision','recall','f1','far','frr')}
    for thr in thresholds:
        y_pred = []
        for dists, true in zip(D, y_te):
            idx = np.argmin(dists)
            y_pred.append(y_tr[idx] if dists[idx] < thr else -1)

        acc  = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred,
                               average='macro', zero_division=0)
        rec  = recall_score(y_te, y_pred,
                             average='macro', zero_division=0)
        f1   = f1_score(y_te, y_pred,
                         average='macro', zero_division=0)
        fa   = sum((p!=t) and (p!=-1)
                   for p,t in zip(y_pred,y_te))
        fr   = sum((p==-1) and (t!=-1)
                   for p,t in zip(y_pred,y_te))

        stats['threshold'].append(thr)
        stats['accuracy'].append(acc)
        stats['precision'].append(prec)
        stats['recall'].append(rec)
        stats['f1'].append(f1)
        stats['far'].append(fa / len(y_te))
        stats['frr'].append(fr / len(y_te))
    return stats


# -------------------
# 2) LSPI implementation
# -------------------
def generate_transitions(distances, y_true, thresholds):
    trans = []
    for dist, true in zip(distances, y_true):
        s = np.array([dist])
        for a_idx, thr in enumerate(thresholds):
            pred = true if dist < thr else -1
            r    = 1 if pred == true else -1
            trans.append((s, a_idx, r, s))
    return trans

def phi(s, a_idx, thresholds):
    onehot = np.zeros(len(thresholds))
    onehot[a_idx] = 1
    return np.concatenate((onehot, s))

def lspi(transitions, thresholds,
         gamma=0.9, max_iter=30, tol=1e-4, ridge=1e-5):
    nA = len(thresholds)
    d  = len(transitions[0][0]) + nA
    theta = np.zeros(d)
    pi    = np.full(len(transitions), nA//2, dtype=int)

    for _ in range(max_iter):
        A = np.eye(d)*ridge
        b = np.zeros(d)
        # policy evaluation
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
        # policy improvement
        for idx, (s,_,_,_) in enumerate(transitions):
            q_vals = [phi(s, i, thresholds).dot(theta)
                      for i in range(nA)]
            pi[idx] = np.argmax(q_vals)

    return theta

def infer_threshold(theta, thresholds, dist):
    s  = np.array([dist])
    q  = [phi(s,i,thresholds).dot(theta) for i in range(len(thresholds))]
    return thresholds[np.argmax(q)]

def evaluate_lspi_policy(X_tr, y_tr, X_te, y_te, thresholds, theta):
    D     = euclidean_distances(X_te, X_tr)
    idx   = np.argmin(D, axis=1)
    dists = D.min(axis=1)
    y_pred = []

    for dist, i in zip(dists, idx):
        thr = infer_threshold(theta, thresholds, dist)
        y_pred.append(y_tr[i] if dist < thr else -1)

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred,
                           average='macro', zero_division=0)
    rec  = recall_score(y_te, y_pred,
                         average='macro', zero_division=0)
    f1   = f1_score(y_te, y_pred,
                     average='macro', zero_division=0)
    fa   = sum((p!=t) and (p!=-1)
               for p,t in zip(y_pred,y_te))
    fr   = sum((p==-1) and (t!=-1)
               for p,t in zip(y_pred,y_te))

    return {
      'accuracy': acc,
      'precision': prec,
      'recall': rec,
      'f1': f1,
      'far': fa/len(y_te),
      'frr': fr/len(y_te)
    }


# -------------------
# 3) Plot & main
# -------------------
def plot_comparison(stats_static, stats_lspi):
    T = stats_static['threshold']
    plt.figure(figsize=(10,6))
    plt.plot(T, stats_static['accuracy'], marker='o', label='Static Acc')
    plt.plot(T, stats_static['precision'], marker='s', label='Static Prec')
    plt.plot(T, stats_static['recall'], marker='^', label='Static Rec')
    plt.plot(T, stats_static['f1'], marker='d', label='Static F1')
    plt.plot(T, stats_static['far'], linestyle='--', label='Static FAR')
    plt.plot(T, stats_static['frr'], linestyle='--', label='Static FRR')

    for m, v in stats_lspi.items():
        plt.hlines(v, T[0], T[-1], linestyles='-.',
                   label=f'LSPI {m.title()}={v:.3f}')

    plt.title("Static vs. LSPI Policy Metrics on PersonDataset")
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.legend(bbox_to_anchor=(1.02,1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    base_dir = r"C:\Users\LUL9HC\Documents\BK\LSPI\PersonDataset"

    # 0) load embeddings + id + pose_group
    X_emb, y_id, pose_gp = load_persondataset(base_dir, n_components=50)

    # 1) random stratified train/test to show LSPI fails under head-pose variation
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_emb, y_id,
        test_size=0.5,
        stratify=y_id,
        random_state=42
    )

    # 2) Static threshold
    thresholds   = np.linspace(0.5, 6.0, 30)
    stats_static = evaluate_thresholds(X_tr, y_tr, X_te, y_te, thresholds)

    # 3) LSPI on same split
    dists       = euclidean_distances(X_te, X_tr).min(axis=1)
    transitions = generate_transitions(dists, y_te, thresholds)
    theta       = lspi(transitions, thresholds)
    stats_lspi  = evaluate_lspi_policy(X_tr, y_tr, X_te, y_te, thresholds, theta)

    # 4) plot
    plot_comparison(stats_static, stats_lspi)


if __name__ == "__main__":
    main()
