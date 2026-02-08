import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mnist1d.data import get_dataset, get_dataset_args
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# 0. CONFIGURAZIONE & SETUP
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configurazione Esperimenti
SEED = 42
LABEL_NOISE = 0.20        # 20% Noise per evidenziare Double Descent
BATCH_SIZE = 128
EPOCHS_DD = 4000          # Epoche per Double Descent (lunghi training)
EPOCHS_GEN = 3000         # Epoche per Generalized NC

# Parametri Modelli
WIDTHS = [2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96] # Asse X per Double Descent
DIMS_GEN = [2, 3, 5, 9, 10, 16, 32] # Dimensioni 'd' per Generalized NC (K=10)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ==========================================
# 1. DATASET (MNIST1D)
# ==========================================
def get_mnist1d_data(label_noise_prob=0.15, batch_size=128):
    args = get_dataset_args()
    data = get_dataset(args, path='./mnist1d_data.pkl', download=True, regenerate=False)
    
    x_train = torch.Tensor(data['x']).float().to(device)
    y_train = torch.LongTensor(data['y']).to(device)
    x_test = torch.Tensor(data['x_test']).float().to(device)
    y_test = torch.LongTensor(data['y_test']).to(device)

    # Label Noise Injection
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
# 2. MODELLO RESNET FLESSIBILE
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
        # Se feature_dim non è specificato, è uguale alla width (caso standard)
        # Se specificato, aggiungiamo un layer di proiezione (caso Generalized NC)
        self.output_dim = feature_dim if feature_dim is not None else width
        
        self.conv_in = nn.Conv1d(1, width, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm1d(width)
        self.relu = nn.ReLU()
        
        self.layer1 = ResNetBlock1D(width)
        self.layer2 = ResNetBlock1D(width)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Proiezione opzionale per Generalized NC
        self.projector = None
        if feature_dim is not None and feature_dim != width:
            self.projector = nn.Linear(width, feature_dim, bias=False)
            
        # Classificatore (Bias=False per Neural Collapse puro)
        self.classifier = nn.Linear(self.output_dim, num_classes, bias=False)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        out = self.conv_in(x)
        out = self.bn_in(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.pool(out)
        h = out.squeeze(2) # [batch, width]
        
        if self.projector:
            h = self.projector(h) # [batch, feature_dim]
            
        logits = self.classifier(h)
        return logits, h

# ==========================================
# 3. METRICHE NEURAL COLLAPSE (NC1-NC4)
# ==========================================
def compute_nc_metrics(model, x_data, y_data):
    """
    Calcola le 4 metriche principali di Neural Collapse.
    Basato su Papyan et al. (2020).
    """
    model.eval()
    K = 10
    with torch.no_grad():
        logits, features = model(x_data)
        W = model.classifier.weight # [K, d]
        
    features = features.cpu().double() # Use double precision for NC metrics
    labels = y_data.cpu()
    W = W.cpu().double()
    
    # 1. Calcolo Medie Globali e di Classe
    global_mean = torch.mean(features, dim=0)
    class_means = []
    for c in range(K):
        indices = (labels == c)
        if indices.sum() == 0:
            class_means.append(torch.zeros_like(global_mean))
        else:
            class_means.append(torch.mean(features[indices], dim=0))
    class_means = torch.stack(class_means) # [K, d]
    
    # --- NC1: Variability Collapse ---
    # Tr(Sigma_W) / Tr(Sigma_B)
    within_scatter = 0
    between_scatter = 0
    
    M = class_means - global_mean # Centered class means
    
    for c in range(K):
        indices = (labels == c)
        if indices.sum() == 0: continue
        
        # Within
        diff_w = features[indices] - class_means[c] # [Nc, d]
        within_scatter += torch.trace(diff_w.T @ diff_w)
        
        # Between
        diff_b = (class_means[c] - global_mean).unsqueeze(1) # [d, 1]
        between_scatter += indices.sum() * torch.trace(diff_b @ diff_b.T)
        
    nc1 = within_scatter / (between_scatter + 1e-9)
    
    # --- NC2: Convergence to Simplex ETF ---
    # Misuriamo la distanza dalla struttura ideale Gram Matrix
    # Ideal ETF Gram: (K/(K-1)) * (I - 1/K 11^T)
    M_centered = class_means - global_mean
    # Normalizziamo le medie per concentrarci solo sugli angoli (Cosine geometry)
    norm_M = torch.norm(M_centered, dim=1, keepdim=True)
    M_normalized = M_centered / (norm_M + 1e-9)
    
    G_empirical = M_normalized @ M_normalized.T # Gram matrix [K, K]
    G_ideal = (torch.eye(K) - 1.0/K) * (K / (K-1.0))
    
    # Metrica: ||G_emp - G_ideal||_F
    nc2 = torch.norm(G_empirical - G_ideal, p='fro').item()
    
    # --- NC3: Self-Duality ---
    # Allineamento tra Classifier Weights W e Class Means M
    # || W/||W|| - M/||M|| ||_F
    W_centered = W - torch.mean(W, dim=0, keepdim=True) # Usually W is centered if bias=False
    
    W_norm = W / (torch.norm(W, dim=1, keepdim=True) + 1e-9)
    M_norm = class_means / (torch.norm(class_means, dim=1, keepdim=True) + 1e-9)
    
    nc3 = torch.norm(W_norm - M_norm, p='fro').item()
    
    # --- NC4: Simplification to NCC ---
    # Mismatch tra predizione rete e predizione Nearest Class Center
    # Calcoliamo accuratezza NCC
    dists = []
    for c in range(K):
        # Distanza Euclidea ||h - mu_c||^2
        d = torch.norm(features - class_means[c], dim=1)**2
        dists.append(d)
    dists = torch.stack(dists, dim=1) # [N, K]
    preds_ncc = torch.argmin(dists, dim=1)
    preds_net = torch.argmax(logits.cpu(), dim=1)
    
    # Frazione di mismatch
    nc4 = (preds_ncc != preds_net).float().mean().item()
    
    return {'NC1': nc1.item(), 'NC2': nc2, 'NC3': nc3, 'NC4': nc4}

# ==========================================
# 4. TRAINING ROUTINE
# ==========================================
def train_model(model, optimizer, train_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    scheduler = None
    if isinstance(optimizer, optim.SGD):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        if scheduler: scheduler.step()
    return model

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
# 5. EXP 1: DOUBLE DESCENT + NC (SGD vs ADAM)
# ==========================================
def run_double_descent_comparison(train_loader, test_loader, x_train, y_train):
    print("\n=== EXP 1: Double Descent & Full NC Metrics (SGD vs Adam) ===")
    
    results = {'SGD': {}, 'Adam': {}}
    
    for opt_name in ['SGD', 'Adam']:
        print(f"\n--- Running with {opt_name} ---")
        metrics = {'width': [], 'train_err': [], 'test_err': [], 
                   'NC1': [], 'NC2': [], 'NC3': [], 'NC4': []}
        
        for w in WIDTHS:
            model = ResNet1D(width=w, num_classes=10).to(device)
            
            if opt_name == 'SGD':
                # SGD favorisce NC e Double Descent pulito
                optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            else:
                # Adam
                optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            
            model = train_model(model, optimizer, train_loader, EPOCHS_DD)
            
            # Calcolo Errori
            tr_err = evaluate_error(model, train_loader)
            te_err = evaluate_error(model, test_loader)
            
            # Calcolo NC Metrics (Sul TRAIN set, dove avviene il fenomeno geometrico)
            nc_res = compute_nc_metrics(model, x_train, y_train)
            
            metrics['width'].append(w)
            metrics['train_err'].append(tr_err)
            metrics['test_err'].append(te_err)
            metrics['NC1'].append(nc_res['NC1'])
            metrics['NC2'].append(nc_res['NC2'])
            metrics['NC3'].append(nc_res['NC3'])
            metrics['NC4'].append(nc_res['NC4'])
            
            print(f"W={w:2d} | TrErr={tr_err:.2f} TeErr={te_err:.3f} | NC1={nc_res['NC1']:.2e} NC2={nc_res['NC2']:.2f}")
            
        results[opt_name] = metrics
        
    return results

# ==========================================
# 6. EXP 2: GENERALIZED NEURAL COLLAPSE
# ==========================================
def run_generalized_nc(train_loader, x_train, y_train):
    print("\n=== EXP 2: Generalized Neural Collapse (Forcing d < K) ===")
    print("Fixed Width = 64 (Over-parameterized), Varying Feature Dim 'd'")
    
    metrics = {'d': [], 'NC1': [], 'NC2': [], 'test_err': []}
    
    # Usiamo SGD perché è noto per indurre NC meglio di Adam
    fixed_width = 64 
    
    for d in DIMS_GEN:
        # Modello con collo di bottiglia 'd' prima del classificatore
        model = ResNet1D(width=fixed_width, feature_dim=d, num_classes=10).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        
        model = train_model(model, optimizer, train_loader, EPOCHS_GEN)
        
        # NC Metrics
        nc_res = compute_nc_metrics(model, x_train, y_train)
        te_err = evaluate_error(model, test_loader) # Globale test_loader
        
        metrics['d'].append(d)
        metrics['NC1'].append(nc_res['NC1'])
        metrics['NC2'].append(nc_res['NC2'])
        metrics['test_err'].append(te_err)
        
        print(f"d={d:2d} | TestErr={te_err:.3f} | NC1={nc_res['NC1']:.2e} | NC2={nc_res['NC2']:.2f}")
        
    return metrics

# ==========================================
# MAIN EXECUTION
# ==========================================
train_loader, test_loader, x_train, y_train = get_mnist1d_data(label_noise_prob=LABEL_NOISE)

# Esegui Esperimento 1
res_dd = run_double_descent_comparison(train_loader, test_loader, x_train, y_train)

# Esegui Esperimento 2
res_gen = run_generalized_nc(train_loader, x_train, y_train)


# ==========================================
# PLOTTING
# ==========================================
# FIGURA 1: Double Descent e NC Metrics (SGD vs Adam)
fig1, axs = plt.subplots(2, 3, figsize=(18, 10))
fig1.suptitle(f'Double Descent & Neural Collapse (Noise {LABEL_NOISE})', fontsize=16)

metrics_list = ['test_err', 'NC1', 'NC2', 'NC3', 'NC4']
titles = ['Test Error (Double Descent)', 'NC1 (Variability)', 'NC2 (Simplex ETF)', 'NC3 (Self-Duality)', 'NC4 (NCC Match)']
scales = ['linear', 'log', 'linear', 'linear', 'linear']

# Plot Test Error (0,0) with Train Error threshold mark
ax = axs[0,0]
for opt in ['SGD', 'Adam']:
    ax.plot(res_dd[opt]['width'], res_dd[opt]['test_err'], 'o-', label=f'{opt} Test')
    # Plot Train Err solo tratteggiato leggero per vedere soglia interpolazione
    ax.plot(res_dd[opt]['width'], res_dd[opt]['train_err'], '--', alpha=0.4, label=f'{opt} Train')

ax.set_title("Test & Train Error")
ax.set_xlabel("Width")
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.legend()
# Evidenzia zona interpolazione (dove train va a 0)
ax.axvspan(8, 14, color='gray', alpha=0.1, label='Interpolation Region')

# Plot NC Metrics
coords = [(0,1), (0,2), (1,0), (1,1)]
for i, metric in enumerate(['NC1', 'NC2', 'NC3', 'NC4']):
    r, c = coords[i]
    ax = axs[r, c]
    for opt in ['SGD', 'Adam']:
        ax.plot(res_dd[opt]['width'], res_dd[opt][metric], 'o-', label=opt)
    
    ax.set_title(titles[i+1])
    ax.set_xscale('log')
    if metric == 'NC1': ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

# Remove empty plot (1,2)
axs[1,2].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.9)


# FIGURA 2: Generalized Neural Collapse
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('Generalized Neural Collapse (Fixed Width=64, Varying Feature Dim d)', fontsize=14)

# NC1 vs d
ax1.plot(res_gen['d'], res_gen['NC1'], 's-', color='tab:green', label='NC1 (Variability)')
ax1.set_xlabel("Feature Dimension d (vs K=10)")
ax1.set_ylabel("NC1 Value (Log)")
ax1.set_yscale('log')
ax1.set_title("NC1: Variability Collapse")
ax1.grid(True, alpha=0.3)
ax1.axvline(x=9, color='red', linestyle='--', label='d = K-1 Threshold')
ax1.legend()

# NC2 vs d
ax2.plot(res_gen['d'], res_gen['NC2'], 's-', color='tab:purple', label='NC2 (ETF Distance)')
ax2.set_xlabel("Feature Dimension d (vs K=10)")
ax2.set_ylabel("NC2 Metric (Frobenius Dist)")
ax2.set_title("NC2: Simplex ETF Geometry")
ax2.grid(True, alpha=0.3)
ax2.axvline(x=9, color='red', linestyle='--', label='d = K-1 Threshold')
ax2.legend()

plt.tight_layout()
plt.show()