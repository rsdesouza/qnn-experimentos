"""
=============================================================================
EXPERIMENTO 1 — Comparação Direta: CNN-VQC vs Baselines Clássicos
Dissertação: Redes Neurais Quânticas para Diagnóstico de Câncer de Pulmão
Autor: Rodolfo da Silva de Souza — CESAR School, 2026
=============================================================================

PRÉ-REQUISITOS (instalar uma vez):
    pip install torch torchvision pennylane pennylane-lightning
    pip install scikit-learn matplotlib pandas numpy tqdm medmnist

COMO EXECUTAR:
    python experimento1.py

SAÍDA:
    - resultados_cenario1.csv   → valores para a Tabela 1
    - curvas_roc_cenario1.png   → Figura 4 (Curvas ROC)
    - log_experimento1.txt      → log completo de treinamento
=============================================================================
"""

import os, time, json, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix)
import pennylane as qml
from tqdm import tqdm
import medmnist
from medmnist import PneumoniaMNIST

# ── Configurações gerais ────────────────────────────────────────────────────
SEED       = 42
N_QUBITS   = [4, 8]      # Cenário 1: testar 4 e 8 qubits
N_LAYERS   = 2           # camadas ansatz
EPOCHS     = 50
BATCH_SIZE = 32
LR         = 1e-3
K_FOLDS    = 5
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR    = '.'         # pasta de saída

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Logger ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUT_DIR, 'log_experimento1.txt'), 'w'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info(f'Device: {DEVICE}')
log.info(f'PennyLane version: {qml.__version__}')
log.info(f'PyTorch version: {torch.__version__}')

# ── Dataset: PneumoniaMNIST (proxy público para classificação binária) ───────
# Substitua por NIH ChestX-ray14 ou LIDC-IDRI para resultados definitivos.
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

log.info('Baixando PneumoniaMNIST...')
train_ds = PneumoniaMNIST(split='train', transform=TRANSFORM, download=True)
test_ds  = PneumoniaMNIST(split='test',  transform=TRANSFORM, download=True)

# Unir treino + teste para validação cruzada k-fold
from torch.utils.data import ConcatDataset
full_ds     = ConcatDataset([train_ds, test_ds])
full_labels = np.array(
    [int(train_ds[i][1].item()) for i in range(len(train_ds))] +
    [int(test_ds[i][1].item())  for i in range(len(test_ds))]
)

log.info(f'Total de amostras: {len(full_ds)} | Classes: {np.unique(full_labels)}')

# ── Circuito Quântico VQC ────────────────────────────────────────────────────
def make_vqc(n_qubits, n_layers):
    dev = qml.device('lightning.qubit', wires=n_qubits)

    @qml.qnode(dev, interface='torch', diff_method='parameter-shift')
    def circuit(inputs, weights):
        # Angle encoding
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        # Ansatz: RY + RZ + CNOT linear
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    weight_shapes = {'weights': (n_layers, n_qubits, 2)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)

# ── Arquitetura CNN-VQC ──────────────────────────────────────────────────────
class CNN_VQC(nn.Module):
    def __init__(self, cnn_backbone, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        # CNN extratora (sem camada final)
        self.cnn = nn.Sequential(*list(cnn_backbone.children())[:-1])
        cnn_out  = cnn_backbone.fc.in_features  # 512 para ResNet-18
        # Redução para dimensão compatível com n_qubits
        self.reducer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_out, n_qubits),
            nn.Tanh()   # escala para [-1, 1] → ângulos via π*x
        )
        # Camada quântica
        self.vqc = make_vqc(n_qubits, n_layers)
        # Saída binária
        self.out = nn.Sigmoid()

    def forward(self, x):
        feat   = self.reducer(self.cnn(x))                        # (batch, n_qubits)
        angles = feat * torch.tensor(np.pi, dtype=torch.float32)  # VQC roda em CPU
        # PennyLane 0.44+ requer processamento amostra a amostra
        q_out  = torch.stack([self.vqc(angles[i].cpu()) for i in range(angles.shape[0])])
        return self.out(q_out.to(x.device).unsqueeze(1))

# ── Modelos Baseline ────────────────────────────────────────────────────────
def get_resnet18():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
    return m

def get_efficientnet():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Sequential(nn.Linear(1280, 1), nn.Sigmoid())
    return m

def get_custom_cnn():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 1), nn.Sigmoid()
    )

# ── Métricas ────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true = np.array(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'accuracy':    round(accuracy_score(y_true, y_pred) * 100, 2),
        'sensitivity': round(tp / (tp + fn + 1e-9) * 100, 2),
        'specificity': round(tn / (tn + fp + 1e-9) * 100, 2),
        'f1_score':    round(f1_score(y_true, y_pred), 4),
        'auc_roc':     round(roc_auc_score(y_true, y_prob), 4),
    }

# ── Loop de treino/avaliação ─────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.float().to(DEVICE).squeeze()
        optimizer.zero_grad()
        preds  = model(imgs).squeeze()
        loss   = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_model(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            probs = model(imgs).squeeze().cpu().numpy()
            all_probs.extend(probs.tolist() if probs.ndim > 0 else [float(probs)])
            all_labels.extend(labels.numpy().reshape(-1).tolist())
    return all_labels, all_probs

# ── Experimento principal ────────────────────────────────────────────────────
def run_experiment(model_name, model_factory, indices, labels):
    log.info(f'\n{"="*60}')
    log.info(f'Modelo: {model_name}')
    log.info(f'{"="*60}')

    skf     = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []
    roc_data = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels[indices])):
        log.info(f'  Fold {fold+1}/{K_FOLDS}')
        train_sub = Subset(full_ds, indices[train_idx])
        val_sub   = Subset(full_ds, indices[val_idx])
        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_sub,   batch_size=BATCH_SIZE, shuffle=False)

        model     = model_factory().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCELoss()

        t0 = time.time()
        for epoch in tqdm(range(EPOCHS), desc=f'    Treinando', leave=False):
            loss = train_epoch(model, train_loader, optimizer, criterion)
        train_time = round(time.time() - t0, 1)

        y_true, y_prob = eval_model(model, val_loader)
        metrics = compute_metrics(y_true, y_prob)
        metrics['train_time_s'] = train_time
        metrics['n_params']     = sum(p.numel() for p in model.parameters())
        fold_metrics.append(metrics)
        roc_data.append((y_true, y_prob))

        log.info(f'    Acc={metrics["accuracy"]}%  F1={metrics["f1_score"]}  '
                 f'AUC={metrics["auc_roc"]}  Tempo={train_time}s')

    # Média e desvio das folds
    avg = {k: round(np.mean([m[k] for m in fold_metrics]), 4) for k in fold_metrics[0]}
    std = {k: round(np.std( [m[k] for m in fold_metrics]), 4) for k in fold_metrics[0]}
    log.info(f'  MÉDIA → Acc={avg["accuracy"]}±{std["accuracy"]}%  '
             f'F1={avg["f1_score"]}±{std["f1_score"]}  '
             f'AUC={avg["auc_roc"]}±{std["auc_roc"]}')
    return avg, std, roc_data

# ── Definir modelos a testar ─────────────────────────────────────────────────
all_indices = np.arange(len(full_ds))
experiments = [
    ('CNN-VQC (4 qubits)', lambda: CNN_VQC(models.resnet18(weights=None), 4, N_LAYERS)),
    ('CNN-VQC (8 qubits)', lambda: CNN_VQC(models.resnet18(weights=None), 8, N_LAYERS)),
    ('ResNet-18',          get_resnet18),
    ('EfficientNet-B0',    get_efficientnet),
    ('CNN Customizada',    get_custom_cnn),
]

# ── Executar todos e coletar resultados ──────────────────────────────────────
results_rows = []
all_roc      = {}

for name, factory in experiments:
    avg, std, roc_data = run_experiment(name, factory, all_indices, full_labels)
    results_rows.append({
        'Modelo':         name,
        'Acurácia (%)':   f'{avg["accuracy"]} ± {std["accuracy"]}',
        'Sensibil. (%)':  f'{avg["sensitivity"]} ± {std["sensitivity"]}',
        'Especif. (%)':   f'{avg["specificity"]} ± {std["specificity"]}',
        'F1-score':       f'{avg["f1_score"]} ± {std["f1_score"]}',
        'AUC-ROC':        f'{avg["auc_roc"]} ± {std["auc_roc"]}',
        'Parâmetros':     avg['n_params'],
        'Tempo treino(s)':avg['train_time_s'],
    })
    all_roc[name] = roc_data

# ── Salvar Tabela 1 em CSV ───────────────────────────────────────────────────
df = pd.DataFrame(results_rows)
csv_path = os.path.join(OUT_DIR, 'resultados_cenario1.csv')
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
log.info(f'\nTabela 1 salva em: {csv_path}')
log.info('\n' + df.to_string(index=False))

# ── Gerar Figura 4 — Curvas ROC ──────────────────────────────────────────────
from sklearn.metrics import roc_curve

plt.figure(figsize=(9, 7))
colors = ['#1F5C99', '#2E75B6', '#E87722', '#C00000', '#70AD47']

for (name, _), color in zip(experiments, colors):
    # Usar última fold para curva ROC ilustrativa
    y_true, y_prob = all_roc[name][-1]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, color=color, linewidth=2,
             label=f'{name} (AUC = {auc_val:.4f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Classificador aleatório')
plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=12)
plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=12)
plt.title('Figura 4 — Curvas ROC: CNN-VQC vs Baselines Clássicos\n'
          'Fonte: Elaborado pelo autor (2026).', fontsize=11)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()

fig_path = os.path.join(OUT_DIR, 'curvas_roc_cenario1.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
log.info(f'Figura 4 salva em: {fig_path}')
log.info('\nExperimento 1 concluído com sucesso.')