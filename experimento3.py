"""
=============================================================================
EXPERIMENTO 3 — Escalabilidade: CNN-VQC vs ResNet-18 por Volume de Dados
                Validação cruzada em DOIS datasets médicos
=============================================================================
Dissertação:
    Redes Neurais Quânticas Aplicadas ao Diagnóstico Assistido de Câncer
    de Pulmão em Bases Massivas de Imagens Médicas

Autor  : Rodolfo da Silva de Souza
Escola : CESAR School — Mestrado em Engenharia de Software (2026)
Orient.: Prof.ª Dr.ª Pamela Thays Lins Bezerra

Objetivo
--------
Avaliar como o tempo de treinamento e o tempo de inferência do CNN-VQC
se comportam em função do volume de dados, comparando com a ResNet-18
como baseline clássico de referência.

Validação cruzada em DOIS datasets médicos com modalidades distintas:
  1. PneumoniaMNIST (raio-X pediátrico) — pneumonia vs normal
  2. BreastMNIST    (ultrassom)         — maligno  vs benigno

A replicação em duas modalidades diferentes fortalece a validade externa
das conclusões: se o padrão de escalabilidade observado se reproduz em
modalidades distintas, o achado não é específico de uma única base.

Hipótese
--------
O custo do VQC por amostra é constante (O(1) por qubit), mas como é
executado sequencialmente por amostra, o tempo total escala linearmente
com o volume — da mesma forma que a CNN clássica. A diferença está na
constante: o VQC tem overhead fixo por amostra muito maior que a GPU
batched. O experimento quantifica empiricamente esta constante e verifica
se o overhead quântico se torna proibitivo em cenários Big Data,
independentemente da modalidade da imagem médica.

Volumes avaliados
-----------------
  PneumoniaMNIST: {1.000, 5.000, 10.000, 50.000} imagens
  BreastMNIST   : {200,   500,   780}             — limitado pelo tamanho da base

Métricas coletadas
------------------
  Tempo de treinamento total (s) por epoch, extrapolado para 50 épocas
  Tempo médio de inferência por imagem (ms)
  Acurácia e AUC-ROC no subconjunto de validação (20% do volume total)

Saídas geradas
--------------
  results/resultados_cenario3.csv               — Tabela 3 da dissertação
  results/escalabilidade_tempo_treino.png       — Figura 5a (tempo treino)
  results/escalabilidade_tempo_inf.png          — Figura 5b (tempo inferência)
  results/escalabilidade_acuracia.png           — Figura 5c (acurácia)
  results/escalabilidade_auc.png                — Figura 5d (AUC-ROC)
  results/escalabilidade_cenario3.png           — Figura 5 combinada (3 painéis)
  results/escalabilidade_cenario3_breast.png    — Figura 5 combinada (BreastMNIST)
  logs/log_experimento3.txt                     — Log completo

Executar:
  python experimento3.py
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import pennylane as qml
from medmnist import PneumoniaMNIST, BreastMNIST

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — CONFIGURAÇÃO GLOBAL
# ══════════════════════════════════════════════════════════════════════════════

SEED        = 42
N_QUBITS    = 8      # configuração do Cenário 1 (melhor AUC-ROC: 0.7931)
N_LAYERS    = 2
EPOCHS      = 10     # reduzido: objetivo é medir tempo/epoch, não convergência
BATCH_SIZE  = 128
LR          = 1e-3
VAL_SPLIT   = 0.20

# Volumes avaliados POR DATASET — adaptados ao tamanho de cada base
VOLUMES_BY_DS = {
    'pneumonia': [1_000, 5_000, 10_000, 50_000],   # base com 5.332 → bootstrap após 5k
    'breast':    [200, 500, 780],                  # base BreastMNIST tem 780 amostras
}

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

# ── Transformação ─────────────────────────────────────────────────────────────
# Mesma transformação para ambos os datasets — ImageNet normalization padrão.
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — CARREGAMENTO DOS DATASETS
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(name: str, log: logging.Logger) -> tuple[ConcatDataset, np.ndarray]:
    """
    Carrega um dos datasets MedMNIST 2D binários e retorna a versão completa
    (treino + teste concatenados) com seus rótulos como array numpy.

    Datasets suportados:
      'pneumonia' — PneumoniaMNIST: raio-X pediátrico, 5.332 amostras
                    Classe 0: Normal | Classe 1: Pneumonia
      'breast'    — BreastMNIST:    ultrassom, 780 amostras
                    Classe 0: Maligno | Classe 1: Benigno

    Parâmetros
    ----------
    name : 'pneumonia' ou 'breast'
    log  : logger configurado

    Retorno
    -------
    (full_ds, labels) — dataset concatenado e array de rótulos
    """
    log.info(f'\nCarregando dataset: {name}')
    if name == 'pneumonia':
        DS = PneumoniaMNIST
        label_map = {0: 'Normal', 1: 'Pneumonia'}
    elif name == 'breast':
        DS = BreastMNIST
        label_map = {0: 'Maligno', 1: 'Benigno'}
    else:
        raise ValueError(f'Dataset desconhecido: {name}')

    tr_ds = DS(split='train', transform=TRANSFORM, download=True)
    te_ds = DS(split='test',  transform=TRANSFORM, download=True)
    full_ds = ConcatDataset([tr_ds, te_ds])

    labels = np.array(
        [int(tr_ds[i][1].item()) for i in range(len(tr_ds))] +
        [int(te_ds[i][1].item()) for i in range(len(te_ds))]
    )
    log.info(f'  Total: {len(full_ds)} | '
             f'Cl.0 ({label_map[0]}): {np.sum(labels==0)} | '
             f'Cl.1 ({label_map[1]}): {np.sum(labels==1)}')
    return full_ds, labels


def build_scaled_dataset(
    base_ds: ConcatDataset,
    base_labels: np.ndarray,
    n_samples: int,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    Constrói índices estratificados de tamanho n_samples a partir do dataset.

    Quando n_samples > len(base_ds), reamostragem COM REPOSIÇÃO (bootstrap)
    estratificada para simular cenários de maior volume sem precisar de
    outro dataset. Isso é válido para avaliar escalabilidade de TEMPO,
    não de generalização.

    Parâmetros
    ----------
    base_ds     : dataset concatenado (treino+teste)
    base_labels : array de rótulos correspondentes
    n_samples   : volume alvo
    rng         : gerador numpy para reprodutibilidade

    Retorno
    -------
    (train_idx, val_idx) — arrays de índices para treino (80%) e validação (20%)
    """
    n_base = len(base_ds)

    if n_samples <= n_base:
        # Subamostragem SEM reposição
        _, sampled_idx = train_test_split(
            np.arange(n_base),
            test_size=n_samples / n_base,
            stratify=base_labels,
            random_state=SEED
        )
    else:
        # Bootstrap COM reposição, mantendo proporção de classes
        class_0 = np.where(base_labels == 0)[0]
        class_1 = np.where(base_labels == 1)[0]
        ratio_1 = len(class_1) / n_base
        n1 = int(n_samples * ratio_1)
        n0 = n_samples - n1
        idx_0 = rng.choice(class_0, size=n0, replace=True)
        idx_1 = rng.choice(class_1, size=n1, replace=True)
        sampled_idx = np.concatenate([idx_0, idx_1])

    sampled_labels = base_labels[sampled_idx]
    train_idx, val_idx = train_test_split(
        sampled_idx,
        test_size=VAL_SPLIT,
        stratify=sampled_labels,
        random_state=SEED
    )
    return train_idx, val_idx


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — CIRCUITO VQC E ARQUITETURA CNN-VQC
# ══════════════════════════════════════════════════════════════════════════════

def make_vqc(n_qubits: int, n_layers: int) -> qml.qnn.TorchLayer:
    """
    VQC com Angle Encoding + ansatz RY+RZ+CNOT linear + medição Pauli-Z.
    Idêntico ao Cenário 1.
    """
    dev = qml.device('lightning.qubit', wires=n_qubits)

    @qml.qnode(dev, interface='torch', diff_method='parameter-shift')
    def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> float:
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    return qml.qnn.TorchLayer(circuit, {'weights': (n_layers, n_qubits, 2)})


class CNN_VQC(nn.Module):
    """
    Arquitetura híbrida CNN-VQC — 8 qubits, 2 camadas (configuração do Cenário 1).

    O VQC é executado SEQUENCIALMENTE por amostra — o PennyLane 0.44 não é
    thread-safe no TorchLayer. O lightning.qubit paraleliza a simulação
    internamente em C++ usando os cores disponíveis.
    """

    def __init__(self, n_qubits: int, n_layers: int):
        super().__init__()
        backbone     = models.resnet18(weights=None)
        self.cnn     = nn.Sequential(*list(backbone.children())[:-1])
        cnn_out      = backbone.fc.in_features
        self.reducer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_out, n_qubits),
            nn.Tanh()
        )
        self.vqc = make_vqc(n_qubits, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles  = self.reducer(self.cnn(x)) * np.pi
        samples = [angles[i].cpu().detach() for i in range(angles.shape[0])]
        q_out   = [self.vqc(s) for s in samples]
        return torch.stack(q_out).to(x.device).unsqueeze(1)


class ResNet18Baseline(nn.Module):
    """
    ResNet-18 com transfer learning para classificação binária.
    Baseline clássico do Cenário 1 (AUC 0,9930).
    """

    def __init__(self):
        super().__init__()
        m    = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(512, 1)
        self.model = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — MEDIÇÃO DE TEMPO E MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════

def measure_training_time(
    model: nn.Module,
    loader: DataLoader,
    n_epochs: int,
    log: logging.Logger
) -> tuple[float, float]:
    """
    Mede tempo total de treinamento e tempo médio por época.

    Usa BCEWithLogitsLoss + AMP para consistência com Cenário 1.
    Retorna (tempo_total_s, tempo_por_epoch_s).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = torch.amp.GradScaler('cuda') if USE_AMP else None

    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        for imgs, lbs in loader:
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
        if (epoch + 1) % 2 == 0:
            log.info(f'      época {epoch+1}/{n_epochs}')

    total_s    = round(time.time() - t0, 1)
    per_epoch  = round(total_s / n_epochs, 2)
    return total_s, per_epoch


def measure_inference_time(
    model: nn.Module,
    loader: DataLoader,
    n_warmup: int = 3
) -> tuple[float, list, list]:
    """
    Mede tempo médio de inferência (ms/imagem) e coleta predições.

    Aquece o modelo com n_warmup batches antes de medir, eliminando
    latência de compilação JIT / cuDNN autotune do primeiro batch.
    """
    model.eval()
    probs, labs = [], []

    # Aquecimento
    for i, (imgs, lbs) in enumerate(loader):
        if i >= n_warmup:
            break
        with torch.no_grad():
            _ = model(imgs.to(DEVICE))

    # Medição
    t0 = time.time()
    n_imgs = 0
    with torch.no_grad():
        for imgs, lbs in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            p = torch.sigmoid(model(imgs)).squeeze().cpu().numpy()
            probs.extend(p.tolist() if p.ndim > 0 else [float(p)])
            labs.extend(lbs.numpy().reshape(-1).tolist())
            n_imgs += imgs.shape[0]

    total_ms     = (time.time() - t0) * 1000
    ms_per_image = round(total_ms / n_imgs, 4)
    return ms_per_image, labs, probs


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — GERAÇÃO DAS FIGURAS
# ══════════════════════════════════════════════════════════════════════════════

CORES   = {'CNN-VQC (8 qubits)': '#2E75B6', 'ResNet-18': '#E87722'}
MARKERS = {'CNN-VQC (8 qubits)': 'o',       'ResNet-18': 's'}

# Coluna do volume — usada pelas figuras (CORREÇÃO do bug do v1)
COL_VOL = 'Volume (imgs)'


def plot_scalability(
    df: pd.DataFrame,
    col_y: str,
    ylabel: str,
    title: str,
    filename: str,
    out_dir: str,
    logy: bool = False,
    fmt: str = '.1f'
) -> None:
    """
    Gera gráfico de linha dupla (CNN-VQC vs ResNet-18) para uma métrica
    em função do volume de dados.

    BUG fix: a coluna do volume nos dicionários de resultado é COL_VOL
    ('Volume (imgs)'), não 'Volume'. A versão anterior do script tentava
    acessar grp['Volume'] e quebrava após horas de execução.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name, grp in df.groupby('Modelo'):
        grp_sorted = grp.sort_values(COL_VOL)
        ax.plot(
            grp_sorted[COL_VOL], grp_sorted[col_y],
            marker=MARKERS.get(model_name, 'o'),
            color=CORES.get(model_name, '#555'),
            linewidth=2, markersize=8, label=model_name
        )
        for _, row in grp_sorted.iterrows():
            ax.annotate(
                f'{row[col_y]:{fmt}}',
                (row[COL_VOL], row[col_y]),
                textcoords='offset points', xytext=(0, 10),
                ha='center', fontsize=8
            )

    ax.set_xlabel('Volume de dados (imagens)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plot] Salvo: {path}')


def plot_combined(df: pd.DataFrame, dataset: str, out_dir: str) -> None:
    """
    Figura 5 combinada (3 subplots) com tempo, inferência e acurácia
    para um dataset específico — versão para a dissertação.
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    configs = [
        ('Tempo treino/epoch (s)', 'Tempo por época (s)', True,  axes[0], 'a', '.0f'),
        ('Inferencia (ms/img)',    'ms / imagem',         False, axes[1], 'b', '.2f'),
        ('Acuracia (%)',           'Acurácia (%)',        False, axes[2], 'c', '.1f'),
    ]
    for col_y, ylabel, logy, ax, letra, fmt in configs:
        for model_name, grp in df.groupby('Modelo'):
            grp_sorted = grp.sort_values(COL_VOL)
            ax.plot(
                grp_sorted[COL_VOL], grp_sorted[col_y],
                marker=MARKERS.get(model_name, 'o'),
                color=CORES.get(model_name, '#555'),
                linewidth=2, markersize=8, label=model_name
            )
            for _, row in grp_sorted.iterrows():
                ax.annotate(
                    f'{row[col_y]:{fmt}}',
                    (row[COL_VOL], row[col_y]),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=7
                )
        ax.set_xlabel('Volume (imagens)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f'({letra})', fontsize=11)
        ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else str(int(x)))
        )
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3, which='both')

    title_ds = 'PneumoniaMNIST' if dataset == 'pneumonia' else 'BreastMNIST'
    fig.suptitle(
        f'Figura 5 — Escalabilidade: CNN-VQC vs ResNet-18 ({title_ds})\n'
        f'Fonte: Elaborado pelo autor (2026).',
        fontsize=11
    )
    plt.tight_layout()
    suffix = '_breast' if dataset == 'breast' else ''
    path = os.path.join(out_dir, f'escalabilidade_cenario3{suffix}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plot] Salvo: {path}')


def generate_all_figures(df_all: pd.DataFrame, out_dir: str) -> None:
    """
    Gera todas as figuras do Cenário 3 a partir do DataFrame consolidado.

    Para o PneumoniaMNIST (dataset principal): figuras individuais 5a-5d
    e a figura combinada principal.
    Para o BreastMNIST (validação): apenas figura combinada — o objetivo
    é confirmar que o padrão se reproduz, não duplicar todos os gráficos.
    """
    df_pneu   = df_all[df_all['Dataset'] == 'pneumonia']
    df_breast = df_all[df_all['Dataset'] == 'breast']

    if not df_pneu.empty:
        print('\nGerando figuras do PneumoniaMNIST...')
        plot_scalability(
            df_pneu, 'Tempo treino/epoch (s)',
            ylabel='Tempo por época (s) — escala logarítmica',
            title='Figura 5a — Tempo de Treinamento por Época vs. Volume\n'
                  'Fonte: Elaborado pelo autor (2026).',
            filename='escalabilidade_tempo_treino.png',
            out_dir=out_dir, logy=True, fmt='.0f'
        )
        plot_scalability(
            df_pneu, 'Inferencia (ms/img)',
            ylabel='Tempo de inferência (ms/imagem)',
            title='Figura 5b — Tempo de Inferência por Imagem vs. Volume\n'
                  'Fonte: Elaborado pelo autor (2026).',
            filename='escalabilidade_tempo_inf.png',
            out_dir=out_dir, logy=False, fmt='.2f'
        )
        plot_scalability(
            df_pneu, 'Acuracia (%)',
            ylabel='Acurácia (%)',
            title='Figura 5c — Acurácia vs. Volume de Dados\n'
                  'Fonte: Elaborado pelo autor (2026).',
            filename='escalabilidade_acuracia.png',
            out_dir=out_dir, logy=False, fmt='.1f'
        )
        plot_scalability(
            df_pneu, 'AUC-ROC',
            ylabel='AUC-ROC',
            title='Figura 5d — AUC-ROC vs. Volume de Dados\n'
                  'Fonte: Elaborado pelo autor (2026).',
            filename='escalabilidade_auc.png',
            out_dir=out_dir, logy=False, fmt='.4f'
        )
        plot_combined(df_pneu, 'pneumonia', out_dir)

    if not df_breast.empty:
        print('\nGerando figura de validação do BreastMNIST...')
        plot_combined(df_breast, 'breast', out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 6 — EXECUÇÃO DA GRADE EXPERIMENTAL POR DATASET
# ══════════════════════════════════════════════════════════════════════════════

def run_grid_for_dataset(
    dataset_name: str,
    full_ds: ConcatDataset,
    labels: np.ndarray,
    volumes: list[int],
    log: logging.Logger
) -> list[dict]:
    """
    Executa a grade volumes × modelos para um dataset específico,
    retornando uma lista de dicionários com os resultados.

    Sistema de cache por configuração — se o cenário for interrompido
    e reiniciado, configurações já concluídas são puladas automaticamente.
    """
    rng = np.random.default_rng(SEED)
    rows = []

    modelos = [
        ('CNN-VQC (8 qubits)', lambda: CNN_VQC(N_QUBITS, N_LAYERS), True),
        ('ResNet-18',          lambda: ResNet18Baseline(),            False),
    ]

    total_runs = len(volumes) * len(modelos)
    run = 0

    for volume in volumes:
        log.info(f'\n{"="*60}')
        log.info(f'DATASET: {dataset_name} | VOLUME: {volume:,} amostras')
        log.info(f'{"="*60}')

        train_idx, val_idx = build_scaled_dataset(full_ds, labels, volume, rng)
        log.info(f'  Treino: {len(train_idx):,} | Validação: {len(val_idx):,}')

        train_sub = Subset(full_ds, train_idx)
        val_sub   = Subset(full_ds, val_idx)

        for model_name, factory, is_quantum in modelos:
            run += 1
            log.info(f'\n[{run}/{total_runs}] {dataset_name} | {model_name} | volume={volume:,}')

            # Cache por configuração — chave inclui dataset
            cache_key = (
                f'{dataset_name}_'
                f'{model_name.replace(" ","_").replace("(","").replace(")","")}'
                f'_v{volume}'
            )
            cache_path = os.path.join(CHECKPOINT_DIR, f'exp3_{cache_key}.npy')
            if os.path.exists(cache_path):
                saved = np.load(cache_path, allow_pickle=True).item()
                rows.append(saved)
                log.info(
                    f'  [cache] t_treino={saved["Tempo treino/epoch (s)"]}s  '
                    f'inf={saved["Inferencia (ms/img)"]}ms  '
                    f'AUC={saved["AUC-ROC"]}'
                )
                continue

            # Batch size: VQC=32 (overhead sequencial); ResNet=128 (GPU batched)
            bs = 32 if is_quantum else BATCH_SIZE
            kw = dict(num_workers=0, pin_memory=(DEVICE.type == 'cuda'))
            train_loader = DataLoader(train_sub, bs, shuffle=True,  **kw)
            val_loader   = DataLoader(val_sub,   bs, shuffle=False, **kw)

            model = factory().to(DEVICE)
            n_params = sum(p.numel() for p in model.parameters())

            log.info(f'  Treinando {EPOCHS} épocas para medir tempo...')
            t_total, t_per_epoch = measure_training_time(model, train_loader, EPOCHS, log)
            log.info(f'  Tempo total ({EPOCHS} epochs): {t_total}s | Por epoch: {t_per_epoch}s')

            # Extrapolação para 50 épocas (configuração padrão)
            t_50epochs = round(t_per_epoch * 50, 1)

            log.info(f'  Medindo inferência ({len(val_idx):,} amostras)...')
            ms_per_img, y_true, y_prob = measure_inference_time(model, val_loader)
            log.info(f'  Inferência: {ms_per_img} ms/imagem')

            y_pred = (np.array(y_prob) >= 0.5).astype(int)
            acc    = round(accuracy_score(y_true, y_pred) * 100, 2)
            try:
                auc = round(roc_auc_score(y_true, y_prob), 4)
            except Exception:
                auc = float('nan')

            log.info(f'  Acc={acc}%  AUC={auc}')

            result = {
                'Dataset':                 dataset_name,
                'Modelo':                  model_name,
                COL_VOL:                   volume,
                'Treino (amostras)':       len(train_idx),
                'Validacao (amostras)':    len(val_idx),
                'Parametros':              n_params,
                'Tempo treino total (s)':  t_total,
                'Tempo treino/epoch (s)':  t_per_epoch,
                'Tempo extrap. 50ep (s)':  t_50epochs,
                'Inferencia (ms/img)':     ms_per_img,
                'Acuracia (%)':            acc,
                'AUC-ROC':                 auc,
            }
            rows.append(result)
            np.save(cache_path, result)

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# GUARD OBRIGATÓRIO NO WINDOWS (Python 3.13)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    os.makedirs(OUT_DIR,        exist_ok=True)
    os.makedirs(LOG_DIR,        exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Logger ────────────────────────────────────────────────────────────────
    log = logging.getLogger('exp3')
    log.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s  %(message)s')
    for h in [
        logging.FileHandler(os.path.join(LOG_DIR, 'log_experimento3.txt'),
                            'w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]:
        h.setFormatter(fmt)
        log.addHandler(h)

    log.info(f'Device     : {DEVICE}  ({GPU_NAME})')
    log.info(f'PyTorch    : {torch.__version__}')
    log.info(f'PennyLane  : {qml.__version__}')
    log.info(f'Python     : {sys.version.split()[0]}')
    log.info(f'AMP        : {"ATIVO" if USE_AMP else "inativo"}')
    log.info(f'CNN-VQC    : {N_QUBITS} qubits | {N_LAYERS} camadas | {EPOCHS} épocas medidas')
    log.info(f'Datasets   : PneumoniaMNIST + BreastMNIST (validação cruzada)')
    log.info(f'Volumes    : pneumonia={VOLUMES_BY_DS["pneumonia"]} | '
             f'breast={VOLUMES_BY_DS["breast"]}')

    # ── Executar grade para os dois datasets ──────────────────────────────────
    all_rows = []

    # Dataset 1 — PneumoniaMNIST (dataset principal, com bootstrap até 50k)
    full_ds, labels = load_dataset('pneumonia', log)
    rows_pneumonia = run_grid_for_dataset(
        'pneumonia', full_ds, labels, VOLUMES_BY_DS['pneumonia'], log
    )
    all_rows.extend(rows_pneumonia)

    # Dataset 2 — BreastMNIST (validação externa, modalidade ultrassom)
    full_ds, labels = load_dataset('breast', log)
    rows_breast = run_grid_for_dataset(
        'breast', full_ds, labels, VOLUMES_BY_DS['breast'], log
    )
    all_rows.extend(rows_breast)

    # ── Tabela 3 — CSV consolidado ────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT_DIR, 'resultados_cenario3.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    log.info(f'\nTabela 3 salva: {csv_path}')
    log.info('\n' + df.to_string(index=False))

    # ── Tabelas separadas por dataset (mais legíveis para a dissertação) ─────
    for ds_name in ['pneumonia', 'breast']:
        df_ds = df[df['Dataset'] == ds_name].drop(columns='Dataset')
        if df_ds.empty:
            continue
        sub_path = os.path.join(OUT_DIR, f'resultados_cenario3_{ds_name}.csv')
        df_ds.to_csv(sub_path, index=False, encoding='utf-8-sig')
        log.info(f'Subtabela salva: {sub_path}')

    # ── Gerar todas as figuras ────────────────────────────────────────────────
    log.info('\nGerando figuras...')
    generate_all_figures(df, OUT_DIR)

    log.info('\nExperimento 3 concluido com sucesso.')