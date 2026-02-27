import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mnist1d.data import get_dataset, get_dataset_args
from torch.utils.data import TensorDataset, DataLoader

#config 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SEED = 1100
LABEL_NOISE = 0.20        # 20% Noise 
BATCH_SIZE = 128

#epochs
EPOCHS_MODEL_WISE = 3000  
EPOCHS_EPOCH_WISE = 15000 
EPOCHS_GEN_NC = 3000      
EPOCHS_CKA = 2000         

# width configurations for model-wise experiment (ResNet & CNN)
WIDTHS = [2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64] 

torch.manual_seed(SEED)
np.random.seed(SEED)

# ==========================================
# 1. DATASET
# ==========================================
def get_mnist1d_data(label_noise_prob=0.15, batch_size=128):
    args = get_dataset_args()
    data = get_dataset(args, path='./mnist1d_data.pkl', download=True, regenerate=False)
    
    x_train = torch.Tensor(data['x']).float().to(device)
    y_train = torch.LongTensor(data['y']).to(device)
    x_test = torch.Tensor(data['x_test']).float().to(device)
    y_test = torch.LongTensor(data['y_test']).to(device)

    # Label Noise
    n_train = len(y_train)
    n_noise = int(n_train * label_noise_prob)
    if n_noise > 0:
        noise_indices = np.random.choice(n_train, n_noise, replace=False)
        new_labels = torch.randint(0, 10, (n_noise,), device=device)
        y_train[noise_indices] = new_labels

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, x_train, y_train

# ==========================================
# 2. MODELS (RESNET & CNN)
# ==========================================
class ResNetBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, width=16, feature_dim=None, num_classes=10):
        super().__init__()
        self.width = width
        self.output_dim = feature_dim if feature_dim is not None else width
        
        self.conv_in = nn.Conv1d(1, width, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm1d(width)
        self.relu = nn.ReLU()
        
        self.layer1 = ResNetBlock1D(width)
        self.layer2 = ResNetBlock1D(width)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.projector = None
        if feature_dim is not None and feature_dim != width:
            self.projector = nn.Linear(width, feature_dim, bias=False)
            
        self.classifier = nn.Linear(self.output_dim, num_classes, bias=False)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        out = self.conv_in(x)
        out = self.bn_in(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.pool(out)
        h = out.squeeze(2) 
        if self.projector: h = self.projector(h)
        logits = self.classifier(h)
        return logits, h

class ResNet1D_WithLayers(nn.Module):
    """ResNet1D che espone le attivazioni intermedie per il CKA."""
    def __init__(self, width=16, num_classes=10):
        super().__init__()
        self.width = width
        self.conv_in = nn.Conv1d(1, width, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm1d(width)
        self.relu = nn.ReLU()
        self.layer1 = ResNetBlock1D(width)
        self.layer2 = ResNetBlock1D(width)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(width, num_classes, bias=False)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        out = self.conv_in(x)
        out = self.bn_in(out)
        h0 = self.relu(out)          # dopo conv_in
        h1 = self.layer1(h0)         # dopo layer1
        h2 = self.layer2(h1)         # dopo layer2
        out = self.pool(h2)
        h_final = out.squeeze(2)     # feature finali
        logits = self.classifier(h_final)
        return logits, h_final, h0, h1, h2

class StandardCNN1D(nn.Module):
    def __init__(self, width=16, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(width), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(width), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(width, width*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(width*2), nn.ReLU(),
            nn.Conv1d(width*2, width*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(width*2), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(width*2, num_classes, bias=False)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        out = self.features(x)
        out = self.pool(out)
        h = out.squeeze(2)
        logits = self.classifier(h)
        return logits, h

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# 3. NC METRICS
# ==========================================
def compute_nc_metrics(model, x_data, y_data):
    model.eval()
    K = 10
    with torch.no_grad():
        logits, features = model(x_data)
        W = model.classifier.weight
        
    features = features.cpu().double()
    labels = y_data.cpu()
    W = W.cpu().double()
    
    global_mean = torch.mean(features, dim=0)
    class_means = []
    for c in range(K):
        indices = (labels == c)
        if indices.sum() == 0: class_means.append(torch.zeros_like(global_mean))
        else: class_means.append(torch.mean(features[indices], dim=0))
    class_means = torch.stack(class_means)
    
    within_scatter = 0; between_scatter = 0
    for c in range(K):
        indices = (labels == c)
        if indices.sum() == 0: continue
        diff_w = features[indices] - class_means[c]
        within_scatter += torch.trace(diff_w.T @ diff_w)
        diff_b = (class_means[c] - global_mean).unsqueeze(1)
        between_scatter += indices.sum() * torch.trace(diff_b @ diff_b.T)
    nc1 = within_scatter / (between_scatter + 1e-9)
    
    M_centered = class_means - global_mean
    M_normalized = M_centered / (torch.norm(M_centered, dim=1, keepdim=True) + 1e-9)
    G_empirical = M_normalized @ M_normalized.T
    G_ideal = (torch.eye(K) - 1.0/K) * (K / (K-1.0))
    nc2 = torch.norm(G_empirical - G_ideal, p='fro').item()
    
    W_norm = W / (torch.norm(W, dim=1, keepdim=True) + 1e-9)
    M_norm = class_means / (torch.norm(class_means, dim=1, keepdim=True) + 1e-9)
    nc3 = torch.norm(W_norm - M_norm, p='fro').item()
    
    dists = []
    for c in range(K):
        d = torch.norm(features - class_means[c], dim=1)**2
        dists.append(d)
    preds_ncc = torch.argmin(torch.stack(dists, dim=1), dim=1)
    preds_net = torch.argmax(logits.cpu(), dim=1)
    nc4 = (preds_ncc != preds_net).float().mean().item()
    
    return {'NC1': nc1.item(), 'NC2': nc2, 'NC3': nc3, 'NC4': nc4}

def train_one_epoch(model, optimizer, loader, criterion):
    model.train()
    for x, y in loader:
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

def evaluate_error(model, loader):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in loader:
            logits, _ = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 1.0 - (correct / total)

# ==========================================
# 4. EXP 1: MODEL-WISE
# ==========================================
def run_model_wise_complete(train_loader, test_loader, x_train, y_train):
    print("\n" + "="*50)
    print("EXP 1: MODEL-WISE (ResNet vs CNN, SGD vs Adam)")
    print("="*50)
    
    results = {} 
    
    configs = [
        ('ResNet', 'Adam', ResNet1D),
        ('ResNet', 'SGD', ResNet1D),
        ('CNN', 'SGD', StandardCNN1D),
        ('CNN', 'Adam', StandardCNN1D)
    ]
    
    for model_name, opt_name, ModelClass in configs:
        key = f"{model_name}_{opt_name}"
        print(f"\n--- Running {key} ---")
        
        metrics = {'params': [], 'test_err': [], 
                   'NC1': [], 'NC2': [], 'NC3': [], 'NC4': []}
        
        for w in WIDTHS:
            model = ModelClass(width=w).to(device)
            n_params = count_parameters(model)
            
            if opt_name == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_MODEL_WISE)
            else:
                optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
                scheduler = None
                
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(EPOCHS_MODEL_WISE):
                train_one_epoch(model, optimizer, train_loader, criterion)
                if scheduler: scheduler.step()
                
            te_err = evaluate_error(model, test_loader)
            nc_res = compute_nc_metrics(model, x_train, y_train)
            
            metrics['params'].append(n_params)
            metrics['test_err'].append(te_err)
            metrics['NC1'].append(nc_res['NC1'])
            metrics['NC2'].append(nc_res['NC2'])
            metrics['NC3'].append(nc_res['NC3'])
            metrics['NC4'].append(nc_res['NC4'])
            
            print(f"W={w:2d} | Prms={n_params:6d} | Err={te_err:.3f} | NC1={nc_res['NC1']:.2e} | NC2={nc_res['NC2']:.2f} | NC3={nc_res['NC3']:.2f} | NC4={nc_res['NC4']:.2f}")
            
        results[key] = metrics
        
    return results

# ==========================================
# 5. EXP 2: EPOCH-WISE
# ==========================================
def run_epoch_wise_dynamics(train_loader, test_loader, x_train, y_train):
    print("\n" + "="*50)
    print("EXP 2: EPOCH-WISE DYNAMICS")
    print("="*50)
    
    configs = [('Critical', 12), ('Large', 64)]
    history = {}
    
    for name, w in configs:
        print(f"Training {name} Model (Width {w})...")
        model = ResNet1D(width=w).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        hist = {'epoch': [], 'test_err': [], 'NC1': []}
        
        for epoch in range(1, EPOCHS_EPOCH_WISE + 1):
            train_one_epoch(model, optimizer, train_loader, criterion)
            
            if epoch < 100 or epoch % 200 == 0:
                te_err = evaluate_error(model, test_loader)
                nc_res = compute_nc_metrics(model, x_train, y_train)
                
                hist['epoch'].append(epoch)
                hist['test_err'].append(te_err)
                hist['NC1'].append(nc_res['NC1'])
                
                if epoch % 1000 == 0:
                    print(f"Ep {epoch}: Err {te_err:.3f}, NC1 {nc_res['NC1']:.2e}")
        
        history[name] = hist
    return history

# ==========================================
# 6. EXP 3: GENERALIZED NC
# ==========================================
def run_generalized_nc(train_loader, x_train, y_train):
    print("\n" + "="*50)
    print("EXP 3: GENERALIZED NC (Varying d)")
    print("="*50)
    
    DIMS = [2, 3, 5, 8, 9, 10, 12, 16]
    metrics = {'d': [], 'NC1': [], 'NC2': []}
    
    for d in DIMS:
        model = ResNet1D(width=64, feature_dim=d).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        
        for epoch in range(EPOCHS_GEN_NC):
            train_one_epoch(model, optimizer, train_loader, nn.CrossEntropyLoss())
            
        nc_res = compute_nc_metrics(model, x_train, y_train)
        
        metrics['d'].append(d)
        metrics['NC1'].append(nc_res['NC1'])
        metrics['NC2'].append(nc_res['NC2'])
        print(f"d={d:2d} | NC1={nc_res['NC1']:.2e} | NC2={nc_res['NC2']:.2f}")
        
    return metrics

# ==========================================
# 7. EXP 4: CKA (Centered Kernel Alignment)
# ==========================================

def centering(K):
    """Centra la matrice kernel: K_c = H K H, con H = I - (1/n) 11^T."""
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype, device=K.device) - torch.ones(n, n, dtype=K.dtype, device=K.device) / n
    return H @ K @ H

def linear_CKA(X, Y):
    """
    Calcola la Linear CKA tra due matrici di attivazioni X (n x p) e Y (n x q).
    
    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    
    Equivalente alla versione con kernel lineare K_X = X X^T, K_Y = Y Y^T,
    ma computazionalmente più efficiente quando p, q << n.
    """
    X = X.double()
    Y = Y.double()
    
    # Centra le colonne
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    numerator = torch.norm(Y.T @ X, p='fro') ** 2
    denominator = torch.norm(X.T @ X, p='fro') * torch.norm(Y.T @ Y, p='fro')
    
    return (numerator / (denominator + 1e-9)).item()

def rbf_CKA(X, Y, sigma=None):
    """
    Calcola la CKA con kernel RBF (Gaussiano).
    
    K_rbf(x_i, x_j) = exp(-||x_i - x_j||^2 / (2 * sigma^2))
    
    Se sigma=None, usa la mediana delle distanze (euristica standard).
    """
    X = X.double()
    Y = Y.double()
    n = X.shape[0]
    
    def rbf_kernel(Z, sig):
        # Distanze al quadrato
        ZZT = Z @ Z.T
        diag = ZZT.diagonal().unsqueeze(1)
        dist_sq = diag + diag.T - 2 * ZZT
        dist_sq = dist_sq.clamp(min=0)
        return torch.exp(-dist_sq / (2 * sig ** 2))
    
    # Stima sigma come mediana delle distanze (solo su un sottoinsieme per efficienza)
    def estimate_sigma(Z):
        sub = Z[:min(500, n)]
        sub_sq = (sub ** 2).sum(dim=1, keepdim=True)
        dist_sq = sub_sq + sub_sq.T - 2 * sub @ sub.T
        dist_sq = dist_sq.clamp(min=0)
        median_dist = dist_sq[dist_sq > 0].sqrt().median()
        return median_dist.item()
    
    sigma_x = sigma if sigma is not None else estimate_sigma(X)
    sigma_y = sigma if sigma is not None else estimate_sigma(Y)
    
    K_X = rbf_kernel(X, sigma_x)
    K_Y = rbf_kernel(Y, sigma_y)
    
    K_Xc = centering(K_X)
    K_Yc = centering(K_Y)
    
    numerator = torch.trace(K_Xc @ K_Yc)
    denominator = torch.norm(K_Xc, p='fro') * torch.norm(K_Yc, p='fro')
    
    return (numerator / (denominator + 1e-9)).item()

def get_layer_activations(model, x_data):
    """
    Estrae le attivazioni da tutti e 4 gli strati di ResNet1D_WithLayers.
    Restituisce un dict: {nome_layer: tensor (n, d)}.
    """
    model.eval()
    with torch.no_grad():
        logits, h_final, h0, h1, h2 = model(x_data)
    
    # h0, h1, h2 sono (n, C, L) -> facciamo global avg pooling -> (n, C)
    activations = {
        'Layer 0 (Stem)':    h0.mean(dim=2).cpu(),
        'Layer 1 (Block 1)': h1.mean(dim=2).cpu(),
        'Layer 2 (Block 2)': h2.mean(dim=2).cpu(),
        'Layer 3 (Feature)': h_final.cpu(),
    }
    return activations

def run_cka_experiment(train_loader, x_train, y_train):
    """
    Esperimento CKA con tre analisi:
    
    A) CKA layer-by-layer tra un modello piccolo (w=8) e uno grande (w=64)
       -> misura la similarità delle rappresentazioni interne
    
    B) CKA tra lo stesso modello a diversi checkpoint di training
       -> misura come le rappresentazioni evolvono nel tempo
    
    C) CKA tra modelli dello stesso tipo ma con seed diversi
       -> misura la riproducibilità delle rappresentazioni
    """
    print("\n" + "="*50)
    print("EXP 4: CKA - Centered Kernel Alignment")
    print("="*50)
    
    # Usiamo un sottoinsieme del training set per efficienza (CKA scala O(n^2))
    N_SUBSET = 500
    idx = torch.randperm(len(y_train))[:N_SUBSET]
    x_sub = x_train[idx]
    y_sub = y_train[idx]
    
    criterion = nn.CrossEntropyLoss()
    
    # --------------------------------------------------
    # A) CKA cross-architettura: piccolo vs grande
    # --------------------------------------------------
    print("\n[A] CKA cross-architettura (w=8 vs w=64) ...")
    
    results_cross = {}
    for w, label in [(8, 'Small (w=8)'), (64, 'Large (w=64)')]:
        model = ResNet1D_WithLayers(width=w).to(device)
        # Usiamo un forward speciale, quindi train_one_epoch va adattato
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        for epoch in range(EPOCHS_CKA):
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                logits, _, _, _, _ = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
        
        acts = get_layer_activations(model, x_sub)
        results_cross[label] = acts
        print(f"  Trained {label}")
    
    # Matrice CKA lineare tra i layer dei due modelli
    layers_small = list(results_cross['Small (w=8)'].values())
    layers_large = list(results_cross['Large (w=64)'].values())
    layer_names = list(results_cross['Small (w=8)'].keys())
    n_layers = len(layer_names)
    
    cka_cross_matrix = np.zeros((n_layers, n_layers))
    for i, ls in enumerate(layers_small):
        for j, ll in enumerate(layers_large):
            cka_cross_matrix[i, j] = linear_CKA(ls, ll)
    
    print("  Matrice CKA cross-architettura (righe=Small, colonne=Large):")
    print("  " + "\t".join([f"L{j}" for j in range(n_layers)]))
    for i, row in enumerate(cka_cross_matrix):
        print(f"  L{i}: " + "\t".join([f"{v:.3f}" for v in row]))
    
    # --------------------------------------------------
    # B) CKA epoch-wise: come evolvono le rappresentazioni
    # --------------------------------------------------
    print("\n[B] CKA epoch-wise (evoluzione durante il training) ...")
    
    CHECKPOINTS = [1, 50, 200, 500, 1000, 2000]
    model_dyn = ResNet1D_WithLayers(width=32).to(device)
    optimizer_dyn = optim.Adam(model_dyn.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Salva le attivazioni al checkpoint 1 (modello non trainato) come riferimento
    checkpoint_acts = {}  # epoch -> {layer_name -> tensor}
    
    epoch_counter = 0
    for target_ep in CHECKPOINTS:
        steps_to_do = target_ep - epoch_counter
        for _ in range(steps_to_do):
            model_dyn.train()
            for x, y in train_loader:
                optimizer_dyn.zero_grad()
                logits, _, _, _, _ = model_dyn(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer_dyn.step()
        epoch_counter = target_ep
        
        acts = get_layer_activations(model_dyn, x_sub)
        checkpoint_acts[target_ep] = acts
        print(f"  Checkpoint ep {target_ep} salvato")
    
    # CKA tra il primo checkpoint e tutti gli altri (per ogni layer)
    cka_epoch_results = {ln: [] for ln in layer_names}
    ref_acts = checkpoint_acts[CHECKPOINTS[0]]  # ep=1 come riferimento
    
    for ep in CHECKPOINTS:
        for ln in layer_names:
            cka_val = linear_CKA(ref_acts[ln], checkpoint_acts[ep][ln])
            cka_epoch_results[ln].append(cka_val)
    
    print("  CKA vs checkpoint iniziale per layer:")
    print("  Epochs: " + str(CHECKPOINTS))
    for ln, vals in cka_epoch_results.items():
        print(f"  {ln}: " + ", ".join([f"{v:.3f}" for v in vals]))
    
    # --------------------------------------------------
    # C) CKA cross-seed: riproducibilità
    # --------------------------------------------------
    print("\n[C] CKA cross-seed (riproducibilità) ...")
    
    SEEDS_CKA = [42, 123, 456]
    seed_models_acts = {}
    
    for s in SEEDS_CKA:
        torch.manual_seed(s)
        m = ResNet1D_WithLayers(width=32).to(device)
        opt = optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        
        for epoch in range(EPOCHS_CKA):
            m.train()
            for x, y in train_loader:
                opt.zero_grad()
                logits, _, _, _, _ = m(x)
                loss = criterion(logits, y)
                loss.backward()
                opt.step()
        
        seed_models_acts[s] = get_layer_activations(m, x_sub)
        print(f"  Trained seed={s}")
    
    torch.manual_seed(SEED)  # ripristina seed originale
    
    # CKA tra tutte le coppie di seed
    seed_pairs = [(SEEDS_CKA[i], SEEDS_CKA[j]) 
                  for i in range(len(SEEDS_CKA)) 
                  for j in range(i+1, len(SEEDS_CKA))]
    
    cka_seed_results = {ln: [] for ln in layer_names}
    pair_labels = [f"S{a}-S{b}" for a, b in seed_pairs]
    
    for s1, s2 in seed_pairs:
        for ln in layer_names:
            cka_val = linear_CKA(seed_models_acts[s1][ln], seed_models_acts[s2][ln])
            cka_seed_results[ln].append(cka_val)
    
    print("  CKA cross-seed per layer (coppie: " + str(pair_labels) + "):")
    for ln, vals in cka_seed_results.items():
        print(f"  {ln}: " + ", ".join([f"{v:.3f}" for v in vals]))
    
    return {
        'cross_matrix':    cka_cross_matrix,
        'layer_names':     layer_names,
        'epoch_results':   cka_epoch_results,
        'checkpoints':     CHECKPOINTS,
        'seed_results':    cka_seed_results,
        'seed_pairs':      pair_labels,
    }

# ==========================================
# MAIN & PLOTTING
# ==========================================
if __name__ == "__main__":
    train_loader, test_loader, x_train, y_train = get_mnist1d_data(label_noise_prob=LABEL_NOISE)
    
    res_model = run_model_wise_complete(train_loader, test_loader, x_train, y_train)
    res_epoch = run_epoch_wise_dynamics(train_loader, test_loader, x_train, y_train)
    res_gen   = run_generalized_nc(train_loader, x_train, y_train)
    res_cka   = run_cka_experiment(train_loader, x_train, y_train)
    
    # --- PLOTTING ---
    fig = plt.figure(figsize=(28, 18), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)
    fig.suptitle("Analisi Completa: Double Descent, Neural Collapse & CKA", fontsize=20)
    
    # ---- Riga 0: Exp 1 ----
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Exp 1: Test Error")
    ax1.set_xlabel("Params (Log)"); ax1.set_ylabel("Error Rate")
    ax1.set_xscale('log'); ax1.grid(True, alpha=0.3)
    for key, val in res_model.items():
        style = 'o-' if 'ResNet' in key else 's-'
        ax1.plot(val['params'], val['test_err'], style, label=key)
    ax1.legend(fontsize=7)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Exp 1: NC1 (Variability)")
    ax2.set_xlabel("Params (Log)"); ax2.set_ylabel("NC1 (Log)")
    ax2.set_xscale('log'); ax2.set_yscale('log'); ax2.grid(True, alpha=0.3)
    for key, val in res_model.items():
        ax2.plot(val['params'], val['NC1'], 'o--', label=key)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("Exp 1: NC2 (Simplex ETF)")
    ax3.set_xlabel("Params (Log)"); ax3.set_ylabel("NC2 Metric")
    ax3.set_xscale('log'); ax3.grid(True, alpha=0.3)
    for key, val in res_model.items():
        ax3.plot(val['params'], val['NC2'], 'o--', label=key)

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_title("Exp 1: NC3 (Self-Duality)")
    ax4.set_xlabel("Params (Log)"); ax4.set_ylabel("NC3 Metric")
    ax4.set_xscale('log'); ax4.grid(True, alpha=0.3)
    for key, val in res_model.items():
        ax4.plot(val['params'], val['NC3'], 'o--', label=key)

    # ---- Riga 1: Exp 1 (NC4) + Exp 2 + Exp 3 ----
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.set_title("Exp 1: NC4 (NCC Mismatch)")
    ax5.set_xlabel("Params (Log)"); ax5.set_ylabel("Mismatch Rate")
    ax5.set_xscale('log'); ax5.grid(True, alpha=0.3)
    for key, val in res_model.items():
        ax5.plot(val['params'], val['NC4'], 'o--', label=key)
    ax5.text(0.5, 0.5, "0 = Perfect Match", transform=ax5.transAxes, ha='center', alpha=0.5)

    ax6 = fig.add_subplot(gs[1, 1])
    ax6.set_title("Exp 2: Epoch-wise Test Error")
    ax6.set_xlabel("Epochs (Log)"); ax6.set_ylabel("Test Error")
    ax6.set_xscale('log'); ax6.grid(True, alpha=0.3)
    for name, hist in res_epoch.items():
        ax6.plot(hist['epoch'], hist['test_err'], label=name)
    ax6.legend()

    ax7 = fig.add_subplot(gs[1, 2])
    ax7.set_title("Exp 3: Gen NC (NC1 vs dim)")
    ax7.set_xlabel("Feature Dim d"); ax7.set_ylabel("NC1 (Log)")
    ax7.set_yscale('log'); ax7.grid(True, alpha=0.3)
    ax7.plot(res_gen['d'], res_gen['NC1'], '^-', color='green')
    ax7.axvline(x=9, color='red', linestyle='--', label='K-1=9')
    ax7.legend()

    ax8 = fig.add_subplot(gs[1, 3])
    ax8.set_title("Exp 3: Gen NC (NC2 vs dim)")
    ax8.set_xlabel("Feature Dim d"); ax8.set_ylabel("NC2")
    ax8.grid(True, alpha=0.3)
    ax8.plot(res_gen['d'], res_gen['NC2'], 'x-', color='purple')
    ax8.axvline(x=9, color='red', linestyle='--', label='K-1=9')
    ax8.legend()

    # ---- Riga 2: Exp 4 (CKA) ----
    layer_names = res_cka['layer_names']
    short_names = [f"L{i}" for i in range(len(layer_names))]

    # CKA A: heatmap cross-architettura
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.set_title("Exp 4A: CKA Cross-Arch\n(righe=Small w=8, col=Large w=64)")
    im = ax9.imshow(res_cka['cross_matrix'], vmin=0, vmax=1, cmap='hot', aspect='auto')
    ax9.set_xticks(range(len(layer_names))); ax9.set_xticklabels(short_names, fontsize=8)
    ax9.set_yticks(range(len(layer_names))); ax9.set_yticklabels(short_names, fontsize=8)
    ax9.set_xlabel("Layer (Large)"); ax9.set_ylabel("Layer (Small)")
    plt.colorbar(im, ax=ax9, fraction=0.046, pad=0.04)
    # Annotazioni numeriche
    for i in range(len(layer_names)):
        for j in range(len(layer_names)):
            ax9.text(j, i, f"{res_cka['cross_matrix'][i,j]:.2f}",
                     ha='center', va='center', fontsize=7,
                     color='white' if res_cka['cross_matrix'][i,j] < 0.5 else 'black')

    # CKA B: evoluzione epoch-wise per layer
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.set_title("Exp 4B: CKA Epoch-wise\n(vs checkpoint ep=1)")
    ax10.set_xlabel("Epochs"); ax10.set_ylabel("CKA")
    ax10.set_xscale('log'); ax10.grid(True, alpha=0.3)
    colors_ep = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    for (ln, vals), c in zip(res_cka['epoch_results'].items(), colors_ep):
        ax10.plot(res_cka['checkpoints'], vals, 'o-', label=ln, color=c)
    ax10.set_ylim(0, 1.05)
    ax10.legend(fontsize=7)

    # CKA C: cross-seed per layer (bar chart)
    ax11 = fig.add_subplot(gs[2, 2])
    ax11.set_title("Exp 4C: CKA Cross-Seed\n(riproducibilità, w=32)")
    ax11.set_xlabel("Layer"); ax11.set_ylabel("CKA medio")
    ax11.grid(True, alpha=0.3, axis='y')
    
    # Media delle coppie per ogni layer
    mean_by_layer = [np.mean(res_cka['seed_results'][ln]) for ln in layer_names]
    std_by_layer  = [np.std(res_cka['seed_results'][ln])  for ln in layer_names]
    x_pos = np.arange(len(layer_names))
    bars = ax11.bar(x_pos, mean_by_layer, yerr=std_by_layer, capsize=4,
                    color=['#e41a1c', '#377eb8', '#4daf4a', '#984ea3'], alpha=0.8)
    ax11.set_xticks(x_pos); ax11.set_xticklabels(short_names, fontsize=8)
    ax11.set_ylim(0, 1.1)
    for bar, val in zip(bars, mean_by_layer):
        ax11.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                  f"{val:.2f}", ha='center', va='bottom', fontsize=8)

    # Legenda layer names
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')
    legend_text = "Legenda Layer (Exp 4):\n\n"
    for i, ln in enumerate(layer_names):
        legend_text += f"L{i}: {ln}\n"
    legend_text += "\nExp 4A: CKA tra layer corrispondenti\n"
    legend_text += "di modelli con width diverso.\n\n"
    legend_text += "Exp 4B: Drift delle rappresentazioni\n"
    legend_text += "nel corso del training.\n\n"
    legend_text += "Exp 4C: Stabilità delle rappresentazioni\n"
    legend_text += "al variare del seed casuale.\n\n"
    legend_text += "CKA=1: rappresentazioni identiche\n"
    legend_text += "CKA=0: rappresentazioni ortogonali"
    ax12.text(0.05, 0.95, legend_text, transform=ax12.transAxes,
              fontsize=9, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig('neural_collapse_cka_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nFigura salvata in neural_collapse_cka_results.png")