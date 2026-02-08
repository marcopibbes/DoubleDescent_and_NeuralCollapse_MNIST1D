import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mnist1d.data import get_dataset, get_dataset_args
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# 0. CONFIGURAZIONE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SEED = 42
LABEL_NOISE = 0.20        # 20% Noise (Cruciale per Double Descent)
BATCH_SIZE = 128

# Epoche (ridotte leggermente per far girare tutto in tempo ragionevole, aumenta se hai GPU potente)
EPOCHS_MODEL_WISE = 3000  
EPOCHS_EPOCH_WISE = 15000 
EPOCHS_GEN_NC = 3000      

# Parametri Larghezza (Width k)
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
# 2. MODELLI (RESNET & CNN)
# ==========================================

# --- Modello A: ResNet1D ---
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
        
        # Proiezione opzionale per Generalized NC
        self.projector = None
        if feature_dim is not None and feature_dim != width:
            self.projector = nn.Linear(width, feature_dim, bias=False)
            
        # Classificatore (Bias=False per teoria NC pura)
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

# --- Modello B: Standard CNN (Bilanciata) ---
class StandardCNN1D(nn.Module):
    def __init__(self, width=16, num_classes=10):
        super().__init__()
        # Crescita [k, k, 2k, 2k] per evitare esplosione parametri
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
        self.classifier = nn.Linear(width*2, num_classes, bias=False) # Bias False per NC

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
# 3. METRICHE NC (1-4)
# ==========================================
def compute_nc_metrics(model, x_data, y_data):
    model.eval()
    K = 10
    with torch.no_grad():
        logits, features = model(x_data)
        W = model.classifier.weight # [K, d]
        
    features = features.cpu().double()
    labels = y_data.cpu()
    W = W.cpu().double()
    
    # 1. Medie
    global_mean = torch.mean(features, dim=0)
    class_means = []
    for c in range(K):
        indices = (labels == c)
        if indices.sum() == 0: class_means.append(torch.zeros_like(global_mean))
        else: class_means.append(torch.mean(features[indices], dim=0))
    class_means = torch.stack(class_means)
    
    # NC1: Variability
    within_scatter = 0
    between_scatter = 0
    for c in range(K):
        indices = (labels == c)
        if indices.sum() == 0: continue
        diff_w = features[indices] - class_means[c]
        within_scatter += torch.trace(diff_w.T @ diff_w)
        diff_b = (class_means[c] - global_mean).unsqueeze(1)
        between_scatter += indices.sum() * torch.trace(diff_b @ diff_b.T)
    nc1 = within_scatter / (between_scatter + 1e-9)
    
    # NC2: ETF
    M_centered = class_means - global_mean
    M_normalized = M_centered / (torch.norm(M_centered, dim=1, keepdim=True) + 1e-9)
    G_empirical = M_normalized @ M_normalized.T
    G_ideal = (torch.eye(K) - 1.0/K) * (K / (K-1.0))
    nc2 = torch.norm(G_empirical - G_ideal, p='fro').item()
    
    # NC3: Self-Duality
    W_norm = W / (torch.norm(W, dim=1, keepdim=True) + 1e-9)
    M_norm = class_means / (torch.norm(class_means, dim=1, keepdim=True) + 1e-9)
    nc3 = torch.norm(W_norm - M_norm, p='fro').item()
    
    # NC4: NCC
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
# 4. EXP 1: MODEL-WISE (CNN & ResNet)
# ==========================================
def run_model_wise_complete(train_loader, test_loader, x_train, y_train):
    print("\n" + "="*50)
    print("EXP 1: MODEL-WISE (ResNet vs CNN, SGD vs Adam)")
    print("="*50)
    
    results = {} 
    
    # Definiamo le configurazioni da testare
    # Tu volevi vedere la differenza tra SGD e Adam per capire NC
    configs = [
        ('ResNet', 'Adam', ResNet1D),
        ('ResNet', 'SGD', ResNet1D),
        ('CNN', 'SGD', StandardCNN1D), # CNN con SGD è il caso classico del paper
        ('CNN', 'Adam', StandardCNN1D)  # CNN con Adam per vedere se NC emerge comunque
    
    ]
    
    for model_name, opt_name, ModelClass in configs:
        key = f"{model_name}_{opt_name}"
        print(f"\n--- Running {key} ---")
        
        metrics = {'params': [], 'test_err': [], 'NC1': [], 'NC2': []}
        
        for w in WIDTHS:
            model = ModelClass(width=w).to(device)
            n_params = count_parameters(model)
            
            # Setup Optimizer
            if opt_name == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_MODEL_WISE)
            else:
                optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
                scheduler = None
                
            criterion = nn.CrossEntropyLoss()
            
            # Train Loop
            for epoch in range(EPOCHS_MODEL_WISE):
                train_one_epoch(model, optimizer, train_loader, criterion)
                if scheduler: scheduler.step()
                
            # Calc Metrics
            te_err = evaluate_error(model, test_loader)
            nc_res = compute_nc_metrics(model, x_train, y_train)
            
            metrics['params'].append(n_params)
            metrics['test_err'].append(te_err)
            metrics['NC1'].append(nc_res['NC1'])
            metrics['NC2'].append(nc_res['NC2'])
            
            print(f"W={w:2d} | Params={n_params} | Err={te_err:.3f} | NC1={nc_res['NC1']:.2e} | NC2={nc_res['NC2']:.2f}")
            
        results[key] = metrics
        
    return results

# ==========================================
# 5. EXP 2: EPOCH-WISE (Solo ResNet per stabilità)
# ==========================================
def run_epoch_wise_dynamics(train_loader, test_loader, x_train, y_train):
    print("\n" + "="*50)
    print("EXP 2: EPOCH-WISE DYNAMICS")
    print("="*50)
    
    # Scegliamo width critica e large basandoci su exp precedenti
    # Width 12 è spesso critica, Width 64 è large
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
# 6. EXP 3: GENERALIZED NC (Feature Dim < K)
# ==========================================
def run_generalized_nc(train_loader, x_train, y_train):
    print("\n" + "="*50)
    print("EXP 3: GENERALIZED NC (Varying d)")
    print("="*50)
    
    DIMS = [2, 3, 5, 8, 9, 10, 12, 16]
    metrics = {'d': [], 'NC1': [], 'NC2': []}
    
    for d in DIMS:
        # Usiamo ResNet width 64 (Large) ma strozziamo output dim
        model = ResNet1D(width=64, feature_dim=d).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        
        # Train
        for epoch in range(EPOCHS_GEN_NC):
            train_one_epoch(model, optimizer, train_loader, nn.CrossEntropyLoss())
            
        nc_res = compute_nc_metrics(model, x_train, y_train)
        
        metrics['d'].append(d)
        metrics['NC1'].append(nc_res['NC1'])
        metrics['NC2'].append(nc_res['NC2'])
        print(f"d={d:2d} | NC1={nc_res['NC1']:.2e} | NC2={nc_res['NC2']:.2f}")
        
    return metrics

# ==========================================
# MAIN & PLOTTING
# ==========================================

if __name__ == "__main__":
    # 1. Carica Dati
    train_loader, test_loader, x_train, y_train = get_mnist1d_data(label_noise_prob=LABEL_NOISE)
    
    # 2. Esegui Esperimenti (Calcola tutto)
    res_model = run_model_wise_complete(train_loader, test_loader, x_train, y_train)
    res_epoch = run_epoch_wise_dynamics(train_loader, test_loader, x_train, y_train)
    res_gen = run_generalized_nc(train_loader, x_train, y_train)
    
    # 3. Visualizzazione Completa (2 righe x 4 colonne)
    fig = plt.figure(figsize=(24, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    
    # --- RIGA 1: EXP 1 (Model-wise: Error + NC1 + NC2 + NC3) ---
    
    # 1. Test Error
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Exp 1: Double Descent (Test Error)")
    ax1.set_xlabel("Params (Log)")
    ax1.set_ylabel("Error Rate")
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    for key, val in res_model.items():
        style = 'o-' if 'ResNet' in key else 's-'
        ax1.plot(val['params'], val['test_err'], style, label=key)
    ax1.legend()

    # 2. NC1 (Variability)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Exp 1: NC1 (Variability Collapse)")
    ax2.set_xlabel("Params (Log)")
    ax2.set_ylabel("NC1 (Log)")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    for key, val in res_model.items():
        style = 'o--' if 'ResNet' in key else 's--'
        ax2.plot(val['params'], val['NC1'], style, label=key)
    
    # 3. NC2 (Simplex ETF)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("Exp 1: NC2 (Simplex ETF)")
    ax3.set_xlabel("Params (Log)")
    ax3.set_ylabel("NC2 Metric")
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    for key, val in res_model.items():
        style = 'o--' if 'ResNet' in key else 's--'
        ax3.plot(val['params'], val['NC2'], style, label=key)

    # 4. NC3 (Self-Duality - Pesi vs Feature)
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_title("Exp 1: NC3 (W aligns with Means)")
    ax4.set_xlabel("Params (Log)")
    ax4.set_ylabel("NC3 Metric")
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)
    for key, val in res_model.items():
        style = 'o--' if 'ResNet' in key else 's--'
        ax4.plot(val['params'], val['NC3'], style, label=key)

    # --- RIGA 2: ALTRE ANALISI (NC4 + Epoch-wise + Generalized) ---

    # 5. NC4 (NCC Match - Model wise)
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.set_title("Exp 1: NC4 (NCC Mismatch)")
    ax5.set_xlabel("Params (Log)")
    ax5.set_ylabel("Mismatch Rate")
    ax5.set_xscale('log')
    ax5.grid(True, alpha=0.3)
    for key, val in res_model.items():
        style = 'o--' if 'ResNet' in key else 's--'
        ax5.plot(val['params'], val['NC4'], style, label=key)
    ax5.text(0.5, 0.5, "0 = Perfect Match with NCC", transform=ax5.transAxes, ha='center', alpha=0.5)

    # 6. Epoch-wise Dynamics (Exp 2)
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.set_title("Exp 2: Epoch-wise (Benign Overfitting)")
    ax6.set_xlabel("Epochs (Log)")
    ax6.set_ylabel("Test Error")
    ax6.set_xscale('log')
    ax6.grid(True, alpha=0.3)
    for name, hist in res_epoch.items():
        ax6.plot(hist['epoch'], hist['test_err'], label=name)
    ax6.legend()

    # 7. Generalized NC1 (Exp 3)
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.set_title("Exp 3: Gen NC (NC1 vs Dim)")
    ax7.set_xlabel("Feature Dim d")
    ax7.set_yscale('log')
    ax7.grid(True, alpha=0.3)
    ax7.plot(res_gen['d'], res_gen['NC1'], '^-', color='green', label='NC1')
    ax7.legend()

    # 8. Generalized NC2 (Exp 3)
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.set_title("Exp 3: Gen NC (NC2 vs Dim)")
    ax8.set_xlabel("Feature Dim d")
    ax8.grid(True, alpha=0.3)
    ax8.plot(res_gen['d'], res_gen['NC2'], 'x-', color='purple', label='NC2')
    ax8.axvline(x=9, color='red', linestyle='--', label='K-1 Bound')
    ax8.legend()

    plt.show()