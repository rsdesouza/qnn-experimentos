"""
=============================================================================
EXPERIMENTO 2 — Impacto do Número de Qubits e Camadas do Ansatz
=============================================================================
Dissertação:
    Redes Neurais Quânticas Aplicadas ao Diagnóstico Assistido de Câncer
    de Pulmão em Bases Massivas de Imagens Médicas

Autor  : Rodolfo da Silva de Souza
Escola : CESAR School — Mestrado em Engenharia de Software (2026)
Orient.: Prof.ª Dr.ª Pamela Thays Lins Bezerra

Objetivo
--------
Análise de sensibilidade da arquitetura CNN-VQC variando dois fatores:
  - Número de qubits  : {4, 8, 12}
  - Camadas do ansatz : {1, 2, 3}

Para cada combinação (grade 3×3 = 9 configurações), avalia-se:
  - Desempenho preditivo: acurácia, F1-score, AUC-ROC
  - Custo paramétrico  : número total de parâmetros treináveis
  - Custo computacional: tempo de treinamento por fold
  - Risco de barren plateau: variância do gradiente (proxy)

O cruzamento sistemático permite identificar a configuração ótima
qubits × camadas antes de escalar para os Cenários 3, 4 e 5.

Hipótese
--------
Existe um ponto de inflexão onde adicionar qubits/camadas não melhora
o desempenho mas aumenta significativamente o custo computacional,
consistente com o fenômeno de barren plateaus reportado por
McClean et al. (2018) para ansatze profundos randomizados.

Saídas geradas
--------------
  results/resultados_cenario2.csv      — Tabela 2 da dissertação
  results/heatmap_auc_cenario2.png     — Figura 3a (AUC por qubits × camadas)
  results/heatmap_tempo_cenario2.png   — Figura 3b (Tempo por qubits × camadas)
  results/heatmap_params_cenario2.png  — Figura 3c (Parâmetros por configuração)
  results/circuito_qubits_grid.png     — Diagrama comparativo dos circuitos
  logs/log_experimento2.txt            — Log completo

Executar:
  python experimento2.py
=============================================================================
"""

import os
import sys
import time
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix
)
import pennylane as qml
from medmnist import PneumoniaMNIST

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — CONFIGURAÇÃO GLOBAL
# ══════════════════════════════════════════════════════════════════════════════

SEED           = 42
EPOCHS         = 30     # reduzido (9 configs × 3 folds = 27 execuções)
BATCH_SIZE     = 64
LR             = 1e-3
K_FOLDS        = 3      # 3-fold para manter custo computacional viável
EARLY_STOP     = 5

# Grade de busca: (n_qubits, n_layers)
QUBIT_GRID  = [4, 8, 12]
LAYERS_GRID = [1, 2, 3]

OUT_DIR        = 'results'
LOG_DIR        = 'logs'
CHECKPOINT_DIR = 'checkpoints'

# ── GPU ───────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE   = torch.device('cuda')
    GPU_NAME = torch.cuda.get_device_name(0)
    USE_AMP  = True
    torch.backends.cudnn.benchmark = True
else:
    DEVICE   = torch.device('cpu')
    GPU_NAME = 'CPU (sem CUDA)'
    USE_AMP  = False

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Transformação de imagens ───────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),   # 64×64 para reduzir custo (9 configs × 3 folds)
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — CIRCUITO VQC PARAMETRIZADO
# ══════════════════════════════════════════════════════════════════════════════

def make_vqc(n_qubits: int, n_layers: int) -> qml.qnn.TorchLayer:
    """
    Constrói VQC com n_qubits qubits e n_layers camadas de ansatz.

    Parâmetros treináveis: n_layers × n_qubits × 2 (RY + RZ por qubit por camada)
    Espaço de Hilbert    : 2^n_qubits amplitudes complexas

    Para n_qubits=12 e n_layers=3: 72 parâmetros quânticos + redução linear.
    """
    dev = qml.device('lightning.qubit', wires=n_qubits)

    @qml.qnode(dev, interface='torch', diff_method='parameter-shift')
    def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> float:
        # Angle encoding: codifica features clássicas como ângulos quânticos
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        # Ansatz: L blocos de rotações + emaranhamento CNOT linear
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)   # θ treinável
                qml.RZ(weights[layer, i, 1], wires=i)   # φ treinável
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    return qml.qnn.TorchLayer(circuit, {'weights': (n_layers, n_qubits, 2)})


def plot_circuit_grid(qubit_grid: list, n_layers: int, out_dir: str) -> None:
    """
    Gera diagrama comparativo dos circuitos para cada configuração de qubits.

    Salva circuito_qubits_grid.png com subplots lado a lado para
    visualização direta das diferenças de profundidade e largura.

    Parâmetros
    ----------
    qubit_grid : lista de configurações de qubits a desenhar
    n_layers   : número de camadas do ansatz (fixo para comparação)
    out_dir    : diretório de saída
    """
    fig, axes = plt.subplots(1, len(qubit_grid),
                             figsize=(6 * len(qubit_grid), 4))
    if len(qubit_grid) == 1:
        axes = [axes]

    for ax, nq in zip(axes, qubit_grid):
        dev = qml.device('lightning.qubit', wires=nq)

        @qml.qnode(dev)
        def circ(inputs, weights):
            for i in range(nq):
                qml.RY(inputs[i], wires=i)
            for layer in range(n_layers):
                for i in range(nq):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                for i in range(nq - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        sample_in  = np.full(nq, np.pi / 4)
        sample_w   = np.full((n_layers, nq, 2), np.pi / 4)
        try:
            _, drawn_ax = qml.draw_mpl(circ, decimals=1)(sample_in, sample_w)
            ax.set_title(f'{nq} qubits · {n_layers} camadas\n'
                         f'Params: {n_layers*nq*2} (VQC)', fontsize=9)
        except Exception:
            ax.text(0.5, 0.5, f'{nq} qubits\n{n_layers} camadas\n'
                    f'(instale pylatexenc\npara visualizar)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_title(f'{nq} qubits · {n_layers} camadas', fontsize=9)

    fig.suptitle(
        f'Figura 3 — Comparativo dos Circuitos VQC ({n_layers} camadas ansatz)\n'
        'Fonte: Elaborado pelo autor (2026), com base em Cerezo et al. (2021).',
        fontsize=10
    )
    plt.tight_layout()
    path = os.path.join(out_dir, 'circuito_qubits_grid.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plot] Salvo: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — ARQUITETURA CNN-VQC PARAMETRIZADA
# ══════════════════════════════════════════════════════════════════════════════

class CNN_VQC(nn.Module):
    """
    CNN-VQC com backbone ResNet-18 reduzido para o Cenário 2.

    Usa ResNet-18 sem pesos pré-treinados (weights=None) para evitar
    viés de transfer learning na análise de sensibilidade.
    O backbone é compartilhado; apenas o VQC varia entre configurações.

    A camada de redução projeta as 512 features para n_qubits ângulos,
    o que varia entre configurações — cada instância é independente.
    """

    def __init__(self, n_qubits: int, n_layers: int):
        super().__init__()
        backbone     = models.resnet18(weights=None)
        self.cnn     = nn.Sequential(*list(backbone.children())[:-1])
        cnn_out      = backbone.fc.in_features   # 512

        # Interface clássica→quântica: 512 → n_qubits ângulos ∈ [-π, π]
        self.reducer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_out, n_qubits),
            nn.Tanh()
        )
        self.vqc = make_vqc(n_qubits, n_layers)
        # Sem Sigmoid: BCEWithLogitsLoss aplica internamente

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles  = self.reducer(self.cnn(x)) * np.pi
        # Sequencial — PennyLane 0.44 não é thread-safe no TorchLayer;
        # lightning.qubit paraleliza a simulação internamente em C++.
        samples = [angles[i].cpu().detach() for i in range(angles.shape[0])]
        q_out   = [self.vqc(s) for s in samples]
        return torch.stack(q_out).to(x.device).unsqueeze(1)


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — MÉTRICAS E UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: list, y_prob: list, thr: float = 0.5) -> dict:
    """
    Calcula métricas clínicas padrão.

    Retorna accuracy (%), sensitivity (%), specificity (%),
    f1_score e auc_roc como dict.
    """
    yp             = (np.array(y_prob) >= thr).astype(int)
    yt             = np.array(y_true)
    tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
    return {
        'accuracy':    round(accuracy_score(yt, yp) * 100, 2),
        'sensitivity': round(tp / (tp + fn + 1e-9) * 100, 2),
        'specificity': round(tn / (tn + fp + 1e-9) * 100, 2),
        'f1_score':    round(f1_score(yt, yp), 4),
        'auc_roc':     round(roc_auc_score(yt, y_prob), 4),
    }


def count_params(model: nn.Module) -> dict:
    """
    Conta parâmetros treináveis separando componentes clássico e quântico.

    Retorno
    -------
    dict com:
      total    : total de parâmetros treináveis
      classical: parâmetros na CNN + reducer
      quantum  : parâmetros no VQC
    """
    quantum   = sum(p.numel() for n, p in model.named_parameters()
                    if 'vqc' in n and p.requires_grad)
    classical = sum(p.numel() for n, p in model.named_parameters()
                    if 'vqc' not in n and p.requires_grad)
    return {
        'total':    classical + quantum,
        'classical': classical,
        'quantum':  quantum,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — TREINO E AVALIAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def make_loaders(train_sub: Subset, val_sub: Subset) -> tuple:
    """
    DataLoaders para Cenário 2.
    Batch 32 fixo (VQC) — num_workers=0 obrigatório no Windows.
    """
    kw = dict(num_workers=0, pin_memory=(DEVICE.type == 'cuda'))
    return (DataLoader(train_sub, 32, shuffle=True,  **kw),
            DataLoader(val_sub,   32, shuffle=False, **kw))


def train_and_eval(
    model: nn.Module,
    tl: DataLoader,
    vl: DataLoader,
    label: str,
    log: logging.Logger
) -> tuple[dict, float]:
    """
    Treina e avalia um modelo com early stopping.

    Retorno
    -------
    (metrics_dict, train_time_seconds)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = torch.amp.GradScaler('cuda') if USE_AMP else None
    best_loss = float('inf')
    patience  = 0
    ckpt      = os.path.join(CHECKPOINT_DIR, f'{label}.pt')

    t0 = time.time()
    for epoch in range(EPOCHS):
        # ── Treino ──
        model.train()
        for imgs, lbs in tl:
            imgs = imgs.to(DEVICE, non_blocking=True)
            lbs  = lbs.float().to(DEVICE, non_blocking=True).squeeze()
            optimizer.zero_grad(set_to_none=True)
            if USE_AMP:
                with torch.amp.autocast('cuda'):
                    loss = criterion(model(imgs).squeeze(), lbs)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(imgs).squeeze(), lbs)
                loss.backward()
                optimizer.step()

        # ── Validação / early stopping ──
        model.eval()
        val_loss = sum(
            criterion(model(imgs.to(DEVICE)).squeeze(),
                      lbs.float().to(DEVICE).squeeze()).item()
            for imgs, lbs in vl
        ) / len(vl)

        if val_loss < best_loss - 1e-4:
            best_loss, patience = val_loss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience += 1
            if patience >= EARLY_STOP:
                log.info(f'      Early stop época {epoch+1} | vl={val_loss:.4f}')
                break

    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))

    train_time = round(time.time() - t0, 1)

    # ── Avaliação ──
    model.eval()
    probs, labs = [], []
    with torch.no_grad():
        for imgs, lbs in vl:
            p = torch.sigmoid(model(imgs.to(DEVICE))).squeeze().cpu().numpy()
            probs.extend(p.tolist() if p.ndim > 0 else [float(p)])
            labs.extend(lbs.numpy().reshape(-1).tolist())

    return compute_metrics(labs, probs), train_time


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 6 — GERAÇÃO DE HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(
    data: np.ndarray,
    qubit_grid: list,
    layers_grid: list,
    title: str,
    filename: str,
    out_dir: str,
    fmt: str = '.3f',
    cmap: str = 'Blues'
) -> None:
    """
    Gera e salva um heatmap de dupla entrada (qubits × camadas).

    Parâmetros
    ----------
    data        : matriz shape (len(qubit_grid), len(layers_grid))
    qubit_grid  : rótulos das linhas (número de qubits)
    layers_grid : rótulos das colunas (número de camadas)
    title       : título do gráfico
    filename    : nome do arquivo PNG
    out_dir     : diretório de saída
    fmt         : formato dos valores nas células
    cmap        : colormap matplotlib
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    df_heat = pd.DataFrame(
        data,
        index=[f'{q} qubits' for q in qubit_grid],
        columns=[f'{l} camada(s)' for l in layers_grid]
    )
    sns.heatmap(
        df_heat, annot=True, fmt=fmt, cmap=cmap,
        linewidths=0.5, ax=ax, annot_kws={'size': 11}
    )
    ax.set_xlabel('Camadas do Ansatz (L)', fontsize=11)
    ax.set_ylabel('Número de Qubits', fontsize=11)
    ax.set_title(title, fontsize=11, pad=12)
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  [heatmap] Salvo: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# GUARD OBRIGATÓRIO NO WINDOWS (Python 3.13)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    os.makedirs(OUT_DIR,        exist_ok=True)
    os.makedirs(LOG_DIR,        exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Logger ────────────────────────────────────────────────────────────────
    log = logging.getLogger('exp2')
    log.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s  %(message)s')
    for h in [
        logging.FileHandler(os.path.join(LOG_DIR, 'log_experimento2.txt'),
                            'w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]:
        h.setFormatter(fmt)
        log.addHandler(h)

    log.info(f'Device     : {DEVICE}  ({GPU_NAME})')
    log.info(f'PyTorch    : {torch.__version__}')
    log.info(f'PennyLane  : {qml.__version__}')
    log.info(f'Python     : {sys.version.split()[0]}')
    log.info(f'AMP        : {"ATIVO" if USE_AMP else "inativo"} | VQC: sequencial')
    log.info(f'Grade      : {len(QUBIT_GRID)} × {len(LAYERS_GRID)} = '
             f'{len(QUBIT_GRID)*len(LAYERS_GRID)} configurações')
    log.info(f'K_FOLDS    : {K_FOLDS} | EPOCHS: {EPOCHS} | EARLY_STOP: {EARLY_STOP}')

    # ── Dataset ───────────────────────────────────────────────────────────────
    log.info('\nCarregando PneumoniaMNIST (64×64 para eficiência)...')
    tr_ds   = PneumoniaMNIST(split='train', transform=TRANSFORM, download=True)
    te_ds   = PneumoniaMNIST(split='test',  transform=TRANSFORM, download=True)
    full_ds = ConcatDataset([tr_ds, te_ds])
    labels  = np.array(
        [int(tr_ds[i][1].item()) for i in range(len(tr_ds))] +
        [int(te_ds[i][1].item()) for i in range(len(te_ds))]
    )
    idx = np.arange(len(full_ds))
    log.info(f'Total: {len(full_ds)} | Cl.0: {np.sum(labels==0)} | Cl.1: {np.sum(labels==1)}')

    # ── Diagrama comparativo dos circuitos ───────────────────────────────────
    log.info('\nGerando diagrama comparativo dos circuitos...')
    plot_circuit_grid(QUBIT_GRID, n_layers=2, out_dir=OUT_DIR)

    # ── Grade de experimentos ─────────────────────────────────────────────────
    # Matrizes para heatmaps — shape (n_qubits, n_layers)
    nq, nl = len(QUBIT_GRID), len(LAYERS_GRID)
    mat_auc      = np.zeros((nq, nl))
    mat_acc      = np.zeros((nq, nl))
    mat_f1       = np.zeros((nq, nl))
    mat_tempo    = np.zeros((nq, nl))
    mat_params_q = np.zeros((nq, nl), dtype=int)
    mat_params_t = np.zeros((nq, nl), dtype=int)

    rows      = []
    skf       = StratifiedKFold(K_FOLDS, shuffle=True, random_state=SEED)
    total_cfg = nq * nl
    cfg_atual = 0

    for qi, n_qubits in enumerate(QUBIT_GRID):
        for li, n_layers in enumerate(LAYERS_GRID):
            cfg_atual += 1
            cfg_label = f'q{n_qubits}_l{n_layers}'
            log.info(f'\n[{cfg_atual}/{total_cfg}] CNN-VQC | '
                     f'{n_qubits} qubits | {n_layers} camada(s)')

            # Cache de configuração completa
            cache = os.path.join(CHECKPOINT_DIR, f'cfg_{cfg_label}_results.npy')
            if os.path.exists(cache):
                saved = np.load(cache, allow_pickle=True).item()
                avg_m, avg_t, params = saved['avg'], saved['time'], saved['params']
                log.info(f'  [cache] AUC={avg_m["auc_roc"]}  Acc={avg_m["accuracy"]}%')
            else:
                fold_metrics = []
                fold_times   = []

                for fold, (ti, vi) in enumerate(skf.split(idx, labels[idx])):
                    log.info(f'  Fold {fold+1}/{K_FOLDS}')
                    tl, vl = make_loaders(
                        Subset(full_ds, idx[ti]),
                        Subset(full_ds, idx[vi])
                    )
                    model  = CNN_VQC(n_qubits, n_layers).to(DEVICE)
                    params = count_params(model)

                    m, t = train_and_eval(
                        model, tl, vl,
                        f'{cfg_label}_fold{fold}', log
                    )
                    fold_metrics.append(m)
                    fold_times.append(t)
                    log.info(f'    Acc={m["accuracy"]}%  F1={m["f1_score"]}  '
                             f'AUC={m["auc_roc"]}  t={t}s')

                avg_m = {k: round(np.mean([x[k] for x in fold_metrics]), 4)
                         for k in fold_metrics[0]}
                std_m = {k: round(np.std( [x[k] for x in fold_metrics]), 4)
                         for k in fold_metrics[0]}
                avg_t = round(np.mean(fold_times), 1)

                np.save(cache, {'avg': avg_m, 'std': std_m,
                                'time': avg_t, 'params': params})

            # Preencher matrizes para heatmaps
            mat_auc[qi, li]      = avg_m['auc_roc']
            mat_acc[qi, li]      = avg_m['accuracy']
            mat_f1[qi, li]       = avg_m['f1_score']
            mat_tempo[qi, li]    = avg_t
            mat_params_q[qi, li] = params['quantum']
            mat_params_t[qi, li] = params['total']

            rows.append({
                'Qubits':          n_qubits,
                'Camadas':         n_layers,
                'Params VQC':      params['quantum'],
                'Params Total':    params['total'],
                'Acuracia (%)':    avg_m['accuracy'],
                'Sensibil. (%)':   avg_m['sensitivity'],
                'Especif. (%)':    avg_m['specificity'],
                'F1-score':        avg_m['f1_score'],
                'AUC-ROC':         avg_m['auc_roc'],
                'Tempo medio (s)': avg_t,
            })

            log.info(f'  MEDIA: AUC={avg_m["auc_roc"]}  '
                     f'Acc={avg_m["accuracy"]}%  Tempo={avg_t}s  '
                     f'Params_VQC={params["quantum"]}')

    # ── Tabela 2 — CSV ────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df = df.sort_values(['AUC-ROC'], ascending=False)
    csv_path = os.path.join(OUT_DIR, 'resultados_cenario2.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    log.info(f'\nTabela 2 salva: {csv_path}')
    log.info('\n' + df.to_string(index=False))

    # Configuração ótima (maior AUC-ROC)
    best = df.iloc[0]
    log.info(f'\nMELHOR CONFIGURAÇÃO: {int(best["Qubits"])} qubits · '
             f'{int(best["Camadas"])} camadas | '
             f'AUC={best["AUC-ROC"]} | Acc={best["Acuracia (%)"]}')

    # ── Heatmaps ─────────────────────────────────────────────────────────────
    log.info('\nGerando heatmaps...')

    plot_heatmap(
        mat_auc, QUBIT_GRID, LAYERS_GRID,
        title='Figura 3a — AUC-ROC por Configuração de Qubits × Camadas\n'
              'Fonte: Elaborado pelo autor (2026).',
        filename='heatmap_auc_cenario2.png',
        out_dir=OUT_DIR, fmt='.4f', cmap='Blues'
    )
    plot_heatmap(
        mat_tempo, QUBIT_GRID, LAYERS_GRID,
        title='Figura 3b — Tempo Médio de Treino (s) por Configuração\n'
              'Fonte: Elaborado pelo autor (2026).',
        filename='heatmap_tempo_cenario2.png',
        out_dir=OUT_DIR, fmt='.0f', cmap='Oranges'
    )
    plot_heatmap(
        mat_params_q, QUBIT_GRID, LAYERS_GRID,
        title='Figura 3c — Parâmetros Treináveis do VQC por Configuração\n'
              'Fonte: Elaborado pelo autor (2026).',
        filename='heatmap_params_cenario2.png',
        out_dir=OUT_DIR, fmt='d', cmap='Greens'
    )

    log.info('Experimento 2 concluido com sucesso.')