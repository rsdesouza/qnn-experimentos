"""
=============================================================================
EXPERIMENTO 1 — Comparação Direta: CNN-VQC vs Baselines Clássicos
=============================================================================
Dissertação:
    Redes Neurais Quânticas Aplicadas ao Diagnóstico Assistido de Câncer
    de Pulmão em Bases Massivas de Imagens Médicas

Autor  : Rodolfo da Silva de Souza
Escola : CESAR School — Mestrado em Engenharia de Software (2026)
Orient.: Prof.ª Dr.ª Pamela Thays Lins Bezerra

Objetivo
--------
Comparar a arquitetura híbrida CNN-VQC com baselines clássicos de Deep
Learning (ResNet-18, EfficientNet-B0, CNN customizada) em classificação
binária de imagens de tórax, avaliando: acurácia, sensibilidade,
especificidade, F1-score e AUC-ROC por validação cruzada 5-fold.

Pipeline do modelo híbrido
--------------------------
  Imagem 224×224
    → CNN (extração de features)
    → Camada de redução linear (512 → n_qubits)
    → Angle Encoding (RY|0⟩ por qubit)
    → Ansatz VQC  (RY + RZ + CNOT linear, L camadas)
    → Medição Pauli-Z  ⟨Z⟩ ∈ [-1, 1]
    → BCEWithLogitsLoss (sigmoid implícito)

Saídas geradas
--------------
  results/resultados_cenario1.csv   — Tabela 1 da dissertação
  results/curvas_roc_cenario1.png   — Figura 4 (Curvas ROC)
  results/circuito_vqc_4q.png       — Figura 2a (diagrama VQC 4 qubits)
  results/circuito_vqc_8q.png       — Figura 2b (diagrama VQC 8 qubits)
  results/circuito_vqc_ascii.txt    — Representação textual dos circuitos
  logs/log_experimento1.txt         — Log completo de treinamento

Otimizações de desempenho
-------------------------
  GPU     : CNN treinada na RTX 5060 com AMP (float16)
  Quântico: VQC executado sequencialmente — lightning.qubit paraleliza
            internamente em C++ usando todos os cores disponíveis
  I/O     : DataLoader com pin_memory para transferência GPU eficiente
  Treino  : Early stopping + checkpoint por fold

Requisitos (instalar UMA vez no venv ativo)
-------------------------------------------
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  pip install pennylane pennylane-lightning scikit-learn matplotlib
  pip install pandas numpy medmnist pylatexenc

Verificar GPU:
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

Executar:
  python experimento1.py
=============================================================================
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # renderização sem display gráfico (headless)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
import pennylane as qml
from medmnist import PneumoniaMNIST

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — HIPERPARÂMETROS E CONFIGURAÇÃO GLOBAL
# ══════════════════════════════════════════════════════════════════════════════

SEED           = 42     # semente global para reprodutibilidade total
N_LAYERS       = 2      # número de camadas do ansatz VQC (L)
EPOCHS         = 50     # máximo de épocas por fold (limitado pelo early stopping)
BATCH_SIZE     = 128    # tamanho de batch para modelos clássicos na GPU
LR             = 1e-3   # taxa de aprendizado Adam
K_FOLDS        = 5      # número de folds para validação cruzada estratificada
EARLY_STOP     = 7      # paciência do early stopping (épocas sem melhora)
NUM_WORKERS    = 0      # OBRIGATÓRIO 0 no Windows — evita RuntimeError de spawn
OUT_DIR        = 'results'    # diretório de saída para CSVs e imagens
LOG_DIR        = 'logs'          # diretório de logs de treinamento
CHECKPOINT_DIR = 'checkpoints'   # diretório de checkpoints por fold (temporário)

# ── Detecção automática de GPU ─────────────────────────────────────────────────
# AMP (Automatic Mixed Precision) converte operações para float16 na GPU,
# dobrando a velocidade e reduzindo o uso de VRAM sem perda de precisão.
if torch.cuda.is_available():
    DEVICE   = torch.device('cuda')
    GPU_NAME = torch.cuda.get_device_name(0)
    USE_AMP  = True
    torch.backends.cudnn.benchmark = True  # otimização automática de kernels cuDNN
else:
    DEVICE   = torch.device('cpu')
    GPU_NAME = 'CPU (sem CUDA — instale PyTorch com cu128)'
    USE_AMP  = False

torch.manual_seed(SEED)
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — PRÉ-PROCESSAMENTO DO DATASET
# ══════════════════════════════════════════════════════════════════════════════

# Transformação padrão ImageNet aplicada a todas as imagens:
#   1. Redimensionamento para 224×224 (entrada padrão ResNet/EfficientNet)
#   2. Conversão para 3 canais (PneumoniaMNIST é grayscale → replicar canal)
#   3. Normalização com média e desvio padrão do ImageNet (pré-treino transfer)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — CIRCUITO QUÂNTICO VARIACIONAL (VQC)
# ══════════════════════════════════════════════════════════════════════════════

def make_vqc(n_qubits: int, n_layers: int) -> qml.qnn.TorchLayer:
    """
    Constrói o Circuito Quântico Variacional (VQC) como uma camada PyTorch.

    Arquitetura do circuito (3 estágios):

    Estágio 1 — Angle Encoding:
        Cada qubit i é inicializado em |0⟩ e rotacionado pelo ângulo xᵢ
        via porta RY. Isso coloca o sistema em superposição, codificando
        o vetor de features clássico como amplitudes quânticas.
        Fórmula: RY(xᵢ)|0⟩ = cos(xᵢ/2)|0⟩ + sin(xᵢ/2)|1⟩

    Estágio 2 — Ansatz parametrizado (L camadas):
        Em cada camada:
          a) Portas RY(θ) e RZ(φ) rotacionam cada qubit em dois eixos
             da esfera de Bloch. θ e φ são os PARÂMETROS TREINÁVEIS.
          b) Portas CNOT em configuração linear (qubit i → qubit i+1)
             introduzem EMARANHAMENTO entre qubits adjacentes, permitindo
             ao modelo capturar correlações não-clássicas entre features.

    Estágio 3 — Medição:
        O valor esperado do observável Pauli-Z no qubit 0 é calculado:
        ⟨Z⟩ = P(|0⟩) - P(|1⟩) ∈ [-1, 1]
        Este escalar é o "logit quântico" que alimenta a BCEWithLogitsLoss.

    Gradientes:
        A diferenciação usa a "parameter-shift rule":
        ∂f/∂θ = [f(θ+π/2) - f(θ-π/2)] / 2
        Isso permite gradientes exatos compatíveis com hardware quântico real.

    Parâmetros
    ----------
    n_qubits : int
        Número de qubits (4 ou 8 neste experimento).
        Define o espaço de Hilbert: 2^n_qubits amplitudes complexas.
    n_layers : int
        Número de camadas do ansatz. Cada camada adiciona
        2 * n_qubits parâmetros treináveis.

    Retorno
    -------
    qml.qnn.TorchLayer
        Camada quântica compatível com nn.Module do PyTorch.
        Parâmetros treináveis: n_layers × n_qubits × 2 (RY + RZ).
    """
    dev = qml.device('lightning.qubit', wires=n_qubits)
    # lightning.qubit: simulador vetorial de alta performance da PennyLane,
    # escrito em C++. Suporta parameter-shift e é o mais rápido para CPU.

    @qml.qnode(dev, interface='torch', diff_method='parameter-shift')
    def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> float:
        """
        Define o circuito quântico como função diferenciável.

        Parâmetros
        ----------
        inputs  : tensor de shape (n_qubits,) — ângulos de encoding (da CNN)
        weights : tensor de shape (n_layers, n_qubits, 2) — parâmetros treináveis

        Retorno
        -------
        float : valor esperado ⟨Z⟩ no qubit 0 ∈ [-1.0, 1.0]
        """
        # ── Angle Encoding ──
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)

        # ── Ansatz: L camadas de RY + RZ + CNOT linear ──
        for layer in range(n_layers):
            # Rotações parametrizadas em cada qubit
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)   # parâmetro θ
                qml.RZ(weights[layer, i, 1], wires=i)   # parâmetro φ
            # Emaranhamento CNOT linear: 0→1, 1→2, ..., (n-2)→(n-1)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        # ── Medição: valor esperado de Pauli-Z no qubit 0 ──
        return qml.expval(qml.PauliZ(0))

    # Encapsula o circuito como camada treinável do PyTorch
    return qml.qnn.TorchLayer(circuit, {'weights': (n_layers, n_qubits, 2)})


def plot_vqc_circuit(n_qubits: int, n_layers: int, out_dir: str) -> None:
    """
    Gera e salva os diagramas do circuito VQC em dois formatos:

    1. PNG (matplotlib) — via qml.draw_mpl()
       Produz diagrama vetorial de alta qualidade com notação padrão
       de circuitos quânticos (portas coloridas, fios horizontais).
       Salvo em: circuito_vqc_{n_qubits}q.png  (300 DPI, pronto para dissertação)

    2. ASCII (texto) — via qml.draw()
       Representação em texto plano do circuito, útil para logs e
       documentação no código. Salvo em: circuito_vqc_ascii.txt

    O circuito é gerado com valores de exemplo (ângulos π/4) apenas
    para visualização — os valores reais são aprendidos durante o treino.

    Parâmetros
    ----------
    n_qubits : int  — número de qubits do circuito a desenhar
    n_layers : int  — número de camadas do ansatz
    out_dir  : str  — diretório onde salvar os arquivos de saída
    """
    dev = qml.device('lightning.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit_to_draw(inputs, weights):
        """Circuito idêntico ao make_vqc, usado exclusivamente para visualização."""
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    # Valores de exemplo para renderizar o diagrama
    sample_inputs  = np.full(n_qubits, np.pi / 4)
    sample_weights = np.full((n_layers, n_qubits, 2), np.pi / 4)

    # ── 1. Diagrama matplotlib (PNG) ──────────────────────────────────────────
    try:
        fig, ax = qml.draw_mpl(circuit_to_draw, decimals=2)(sample_inputs, sample_weights)
        fig.suptitle(
            f'Figura 2 — Circuito VQC: {n_qubits} qubits, {n_layers} camadas ansatz\n'
            f'Angle Encoding → RY+RZ+CNOT (×{n_layers}) → Medição Pauli-Z\n'
            f'Fonte: Elaborado pelo autor (2026), com base em Cerezo et al. (2021).',
            fontsize=9, y=1.02
        )
        fig.tight_layout()
        png_path = os.path.join(out_dir, f'circuito_vqc_{n_qubits}q.png')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'  [VQC plot] Salvo: {png_path}')
    except Exception as e:
        print(f'  [VQC plot] Aviso — draw_mpl falhou ({e}). Instalando pylatexenc pode resolver.')

    # ── 2. Representação ASCII (TXT) ──────────────────────────────────────────
    ascii_repr = qml.draw(circuit_to_draw, decimals=2)(sample_inputs, sample_weights)
    ascii_path = os.path.join(out_dir, 'circuito_vqc_ascii.txt')

    # Modo 'a' para acumular representações de 4q e 8q no mesmo arquivo
    with open(ascii_path, 'a', encoding='utf-8') as f:
        f.write(f'\n{"="*70}\n')
        f.write(f'VQC — {n_qubits} qubits | {n_layers} camadas ansatz\n')
        f.write(f'Parâmetros treináveis: {n_layers * n_qubits * 2} '
                f'(RY+RZ por qubit por camada)\n')
        f.write(f'Espaço de Hilbert: 2^{n_qubits} = {2**n_qubits} amplitudes\n')
        f.write(f'{"="*70}\n\n')
        f.write(ascii_repr)
        f.write('\n')
    print(f'  [VQC plot] ASCII salvo: {ascii_path}')


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — ARQUITETURA HÍBRIDA CNN-VQC
# ══════════════════════════════════════════════════════════════════════════════

class CNN_VQC(nn.Module):
    """
    Arquitetura híbrida quântico-clássica para classificação binária.

    Combina uma CNN clássica para extração de features visuais com um
    Circuito Quântico Variacional (VQC) para classificação no espaço
    de Hilbert de dimensão 2^n_qubits.

    Fluxo de dados (forward pass)
    -----------------------------
    1. CNN extratora (ResNet-18 sem camada final):
       Imagem (3×224×224) → vetor de 512 features

    2. Camada de redução (clássica):
       512 → n_qubits valores via Linear + Tanh
       Tanh limita saída a [-1, 1], escalada para ângulos [-π, π]

    3. Angle Encoding (quântico):
       Cada valor se torna o ângulo RY(xᵢ)|0⟩ de um qubit

    4. Ansatz VQC (quântico, parâmetros treináveis):
       RY(θ) + RZ(φ) + CNOT linear, repetido L vezes

    5. Medição Pauli-Z:
       ⟨Z⟩ ∈ [-1, 1] → logit para BCEWithLogitsLoss

    Nota sobre thread safety e paralelismo
    ----------------------------------------
    O PennyLane 0.44 tem validação estrita do estado do qnode: chamadas
    simultâneas ao mesmo TorchLayer de threads diferentes corrompem o
    estado interno do circuito, causando QuantumFunctionError.

    Por isso, as amostras do batch são processadas SEQUENCIALMENTE pelo VQC.
    O paralelismo ocorre naturalmente dentro do lightning.qubit: o simulador
    C++ utiliza todos os cores disponíveis (Ryzen 7 5700X) para a simulação
    do estado quântico de cada amostra individualmente.

    Parâmetros
    ----------
    backbone  : nn.Module — CNN pré-treinada (ResNet-18 recomendado)
    n_qubits  : int — número de qubits (4 ou 8)
    n_layers  : int — camadas do ansatz VQC
    """

    def __init__(self, backbone: nn.Module, n_qubits: int, n_layers: int):
        super().__init__()
        self.n_qubits = n_qubits

        # Remove a última camada (fc) da CNN — apenas extração de features
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        cnn_out  = backbone.fc.in_features  # 512 para ResNet-18

        # Camada de interface clássica → quântica:
        # Projeta 512 features para n_qubits ângulos em [-π, π]
        self.reducer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_out, n_qubits),
            nn.Tanh()   # Tanh ∈ [-1,1]; multiplicado por π → ângulos ∈ [-π, π]
        )

        # Camada quântica treinável (parâmetros: n_layers × n_qubits × 2)
        self.vqc = make_vqc(n_qubits, n_layers)

        # Nota: Sigmoid REMOVIDO — BCEWithLogitsLoss aplica internamente.
        # Isso é necessário para compatibilidade com AMP (float16).

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da arquitetura híbrida.

        Parâmetro
        ---------
        x : tensor de shape (batch, 3, 224, 224) na GPU

        Retorno
        -------
        tensor de shape (batch, 1) com logits quânticos (sem sigmoid)
        """
        # Parte clássica: extração de features na GPU (float16 com AMP)
        angles = self.reducer(self.cnn(x)) * np.pi   # (batch, n_qubits)

        # Parte quântica: amostras processadas SEQUENCIALMENTE pelo VQC.
        # Não usar ThreadPoolExecutor: o PennyLane 0.44 não é thread-safe
        # quando múltiplas threads chamam o mesmo TorchLayer simultaneamente
        # (QuantumFunctionError: "All measurements must be returned in order").
        # O lightning.qubit já paraleliza a simulação internamente em C++.
        samples = [angles[i].cpu().detach() for i in range(angles.shape[0])]
        q_out   = [self.vqc(s) for s in samples]

        # Reconstrói tensor e move de volta para GPU
        return torch.stack(q_out).to(x.device).unsqueeze(1)


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — BASELINES CLÁSSICOS
# ══════════════════════════════════════════════════════════════════════════════

def get_resnet18() -> nn.Module:
    """
    ResNet-18 com pesos ImageNet para classificação binária.

    Transfer learning: todos os pesos são inicializados com ImageNet
    e ajustados (fine-tuning completo) durante o treinamento.
    A camada fc original (1000 classes) é substituída por Linear(512→1).

    Saída: logit escalar (sem Sigmoid — compatível com BCEWithLogitsLoss + AMP).
    """
    m    = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(512, 1)   # logit
    return m


def get_efficientnet() -> nn.Module:
    """
    EfficientNet-B0 com pesos ImageNet para classificação binária.

    Arquitetura baseada em MBConv (Mobile Inverted Bottleneck) com
    compound scaling. Mais eficiente que ResNet-18 em parâmetros/acurácia.
    Saída: 1280 features → Linear(1280→1) logit.
    """
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(1280, 1)   # logit
    return m


def get_custom_cnn() -> nn.Module:
    """
    CNN customizada treinada do zero (sem transfer learning).

    Arquitetura:
        Conv(3→32, 3×3) → ReLU → MaxPool(2×2)   # 112×112
        Conv(32→64, 3×3) → ReLU → MaxPool(2×2)  # 56×56
        Conv(64→128, 3×3) → ReLU → AdaptiveAvgPool(1×1)  # 1×1
        Flatten → Linear(128→1)  # logit

    Serve como baseline mínimo para verificar se o ganho das arquiteturas
    maiores (ResNet, EfficientNet, CNN-VQC) é estatisticamente relevante.
    """
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 1)   # logit
    )


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 6 — MÉTRICAS DE AVALIAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: list, y_prob: list, thr: float = 0.5) -> dict:
    """
    Calcula as cinco métricas clínicas padrão para diagnóstico assistido.

    Parâmetros
    ----------
    y_true : lista de rótulos verdadeiros {0, 1}
    y_prob : lista de probabilidades preditas ∈ [0, 1]
    thr    : limiar de decisão (padrão 0.5)

    Retorno
    -------
    dict com as métricas (valores em %, exceto F1-score e AUC-ROC):

    - accuracy    : (TP+TN)/(TP+TN+FP+FN) — proporção de predições corretas
    - sensitivity : TP/(TP+FN) — recall; capacidade de detectar casos positivos
                    (crítico em diagnóstico — minimiza falsos negativos)
    - specificity : TN/(TN+FP) — capacidade de rejeitar casos negativos
    - f1_score    : média harmônica de precisão e recall
    - auc_roc     : área sob a curva ROC ∈ [0,1]; 1.0 = classificador perfeito
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


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 7 — DATALOADERS
# ══════════════════════════════════════════════════════════════════════════════

def make_loaders(
    train_sub: Subset,
    val_sub: Subset,
    is_quantum: bool = False
) -> tuple[DataLoader, DataLoader]:
    """
    Cria DataLoaders otimizados para treino e validação.

    Configurações de otimização:
    - num_workers=0  : obrigatório no Windows com Python 3.13 para evitar
                       RuntimeError de spawn no multiprocessing
    - pin_memory=True: aloca memória RAM "pinned" (não paginável), acelerando
                       transferências para GPU via DMA

    Parâmetros
    ----------
    train_sub  : subconjunto de treino do fold atual
    val_sub    : subconjunto de validação do fold atual
    is_quantum : se True, usa batch_size=32 (VQC é lento por amostra);
                 se False, usa BATCH_SIZE=128 (GPU processa em lote eficientemente)

    Retorno
    -------
    (train_loader, val_loader)
    """
    bs = 32 if is_quantum else BATCH_SIZE
    kw = dict(num_workers=0, pin_memory=(DEVICE.type == 'cuda'))
    return (
        DataLoader(train_sub, bs, shuffle=True,  **kw),
        DataLoader(val_sub,   bs, shuffle=False, **kw),
    )


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 8 — LOOP DE TREINAMENTO
# ══════════════════════════════════════════════════════════════════════════════

def train_fold(
    model: nn.Module,
    tl: DataLoader,
    vl: DataLoader,
    safe_name: str,
    fold: int,
    log: logging.Logger
) -> float:
    """
    Treina o modelo por um fold com AMP, early stopping e checkpoint.

    Estratégias utilizadas
    ----------------------
    AMP (Automatic Mixed Precision):
        Usa float16 nas operações de forward e backward na GPU,
        dobrando a velocidade e reduzindo uso de VRAM.
        GradScaler previne underflow de gradientes em float16.

    BCEWithLogitsLoss:
        Combina Sigmoid + BCE em uma operação numericamente estável,
        obrigatória para compatibilidade com AMP (BCELoss sozinha falha
        com float16 por instabilidade em valores extremos).

    Early Stopping:
        Monitora val_loss. Se não melhorar por EARLY_STOP épocas
        consecutivas, interrompe e carrega o melhor modelo salvo.
        Evita overfitting e economiza tempo de execução.

    Checkpoint:
        O melhor modelo (menor val_loss) é salvo em disco a cada fold.
        Se o experimento for interrompido, o próximo fold é retomado
        automaticamente pelo cache de métricas.

    Parâmetros
    ----------
    model     : modelo a treinar (já movido para DEVICE)
    tl        : DataLoader de treino
    vl        : DataLoader de validação
    safe_name : nome seguro do modelo (sem espaços/parênteses) para o checkpoint
    fold      : índice do fold atual (0-4)
    log       : logger configurado

    Retorno
    -------
    float : tempo total de treinamento do fold em segundos
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = torch.amp.GradScaler('cuda') if USE_AMP else None
    best_loss = float('inf')
    patience  = 0
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{safe_name}_fold{fold}.pt')

    t0 = time.time()
    for epoch in range(EPOCHS):

        # ── Fase de treinamento ──────────────────────────────────────────────
        model.train()
        for imgs, lbs in tl:
            imgs = imgs.to(DEVICE, non_blocking=True)
            lbs  = lbs.float().to(DEVICE, non_blocking=True).squeeze()
            optimizer.zero_grad(set_to_none=True)  # set_to_none libera memória

            if USE_AMP:
                # autocast converte ops elegíveis para float16 automaticamente
                with torch.amp.autocast('cuda'):
                    loss = criterion(model(imgs).squeeze(), lbs)
                scaler.scale(loss).backward()   # gradientes escalados
                scaler.step(optimizer)          # desescala e atualiza pesos
                scaler.update()                 # ajusta fator de escala
            else:
                loss = criterion(model(imgs).squeeze(), lbs)
                loss.backward()
                optimizer.step()

        # ── Fase de validação (early stopping) ──────────────────────────────
        model.eval()
        val_loss = sum(
            criterion(model(imgs.to(DEVICE)).squeeze(),
                      lbs.float().to(DEVICE).squeeze()).item()
            for imgs, lbs in vl
        ) / len(vl)

        if val_loss < best_loss - 1e-4:
            best_loss, patience = val_loss, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1
            if patience >= EARLY_STOP:
                log.info(f'    Early stop na época {epoch+1} | val_loss={val_loss:.4f}')
                break

    # Carrega o melhor modelo salvo (menor val_loss)
    if os.path.exists(ckpt_path):
        model.load_state_dict(
            torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        )
    return round(time.time() - t0, 1)


def eval_model(model: nn.Module, loader: DataLoader) -> tuple[list, list]:
    """
    Avalia o modelo e retorna rótulos verdadeiros e probabilidades preditas.

    Nota: os modelos retornam logits (sem Sigmoid). O Sigmoid é aplicado
    aqui explicitamente para converter logits em probabilidades ∈ [0, 1],
    necessário para calcular AUC-ROC e outras métricas baseadas em threshold.

    Parâmetros
    ----------
    model  : modelo treinado
    loader : DataLoader de validação

    Retorno
    -------
    (y_true, y_prob) — listas de rótulos e probabilidades
    """
    model.eval()
    probs, labs = [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            # torch.sigmoid converte logits → probabilidades ∈ [0, 1]
            p = torch.sigmoid(model(imgs.to(DEVICE))).squeeze().cpu().numpy()
            probs.extend(p.tolist() if p.ndim > 0 else [float(p)])
            labs.extend(lbs.numpy().reshape(-1).tolist())
    return labs, probs


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 9 — LOOP PRINCIPAL DO EXPERIMENTO
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    name: str,
    factory,
    indices: np.ndarray,
    labels: np.ndarray,
    full_ds: ConcatDataset,
    is_quantum: bool,
    log: logging.Logger
) -> tuple[dict, dict, list]:
    """
    Executa validação cruzada K-fold estratificada para um modelo.

    Validação cruzada estratificada garante que a proporção de classes
    (1448 negativos / 3884 positivos) seja mantida em cada fold,
    evitando viés de avaliação em datasets desbalanceados.

    Sistema de cache por fold:
        Cada fold concluído salva suas métricas em disco (.npy).
        Se o experimento for interrompido e reiniciado, os folds
        já calculados são carregados do cache, evitando reprocessamento.

    Parâmetros
    ----------
    name       : nome do modelo (para log e cache)
    factory    : função lambda que instancia o modelo (chamada por fold)
    indices    : array com índices de todas as amostras
    labels     : array com rótulos correspondentes
    full_ds    : dataset completo (treino + teste concatenados)
    is_quantum : controla batch_size e DataLoader (True para CNN-VQC)
    log        : logger configurado

    Retorno
    -------
    avg     : dict com médias das métricas sobre os K folds
    std     : dict com desvios padrão das métricas
    roc_data: lista de (y_true, y_prob) por fold — para curvas ROC
    """
    safe = name.replace(' ', '_').replace('(', '').replace(')', '')
    log.info(f'\n{"="*60}\nModelo: {name}\n{"="*60}')

    skf      = StratifiedKFold(K_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []
    roc_data     = []

    for fold, (ti, vi) in enumerate(skf.split(indices, labels[indices])):

        # ── Verificar cache ──────────────────────────────────────────────────
        cache_path = os.path.join(CHECKPOINT_DIR, f'{safe}_fold{fold}_metrics.npy')
        if os.path.exists(cache_path):
            saved = np.load(cache_path, allow_pickle=True).item()
            fold_metrics.append(saved['metrics'])
            roc_data.append(saved['roc'])
            log.info(f'  Fold {fold+1}/{K_FOLDS} [cache] Acc={saved["metrics"]["accuracy"]}%')
            continue

        # ── Treinar e avaliar fold ───────────────────────────────────────────
        log.info(f'  Fold {fold+1}/{K_FOLDS}')
        tl, vl     = make_loaders(
            Subset(full_ds, indices[ti]),
            Subset(full_ds, indices[vi]),
            is_quantum
        )
        model      = factory().to(DEVICE)
        train_time = train_fold(model, tl, vl, safe, fold, log)
        y_true, y_prob = eval_model(model, vl)
        metrics        = compute_metrics(y_true, y_prob)
        metrics['train_time_s'] = train_time
        metrics['n_params']     = sum(p.numel() for p in model.parameters())

        fold_metrics.append(metrics)
        roc_data.append((y_true, y_prob))

        # Salva cache do fold concluído
        np.save(cache_path, {'metrics': metrics, 'roc': (y_true, y_prob)})
        log.info(
            f'    Acc={metrics["accuracy"]}%  F1={metrics["f1_score"]}  '
            f'AUC={metrics["auc_roc"]}  Tempo={train_time}s'
        )

    # ── Estatísticas agregadas sobre os K folds ──────────────────────────────
    avg = {k: round(np.mean([m[k] for m in fold_metrics]), 4) for k in fold_metrics[0]}
    std = {k: round(np.std( [m[k] for m in fold_metrics]), 4) for k in fold_metrics[0]}
    log.info(
        f'  MEDIA: Acc={avg["accuracy"]}+-{std["accuracy"]}%  '
        f'F1={avg["f1_score"]}+-{std["f1_score"]}  '
        f'AUC={avg["auc_roc"]}+-{std["auc_roc"]}'
    )
    return avg, std, roc_data


# ══════════════════════════════════════════════════════════════════════════════
# GUARD OBRIGATÓRIO NO WINDOWS (Python 3.13)
#
# No Windows, o módulo multiprocessing usa "spawn" (não "fork") para criar
# subprocessos. Sem o guard, cada worker do DataLoader importa este módulo
# completo como script principal, disparando os experimentos recursivamente
# e causando o RuntimeError de spawn.
#
# Com num_workers=0 (valor atual), o guard não é estritamente necessário,
# mas é mantido como boa prática e proteção contra mudanças futuras.
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    os.makedirs(OUT_DIR,        exist_ok=True)
    os.makedirs(LOG_DIR,        exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Logger ────────────────────────────────────────────────────────────────
    log = logging.getLogger('exp1')
    log.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s  %(message)s')
    for handler in [
        logging.FileHandler(
            os.path.join(LOG_DIR, 'log_experimento1.txt'), 'w', encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout)
    ]:
        handler.setFormatter(fmt)
        log.addHandler(handler)

    log.info(f'Device     : {DEVICE}  ({GPU_NAME})')
    log.info(f'PyTorch    : {torch.__version__}')
    log.info(f'PennyLane  : {qml.__version__}')
    log.info(f'Python     : {sys.version.split()[0]}')
    log.info(f'AMP        : {"ATIVO (float16)" if USE_AMP else "inativo (CPU)"}')
    log.info(f'Batch size : {BATCH_SIZE} | VQC: sequencial (lightning.qubit paralelo interno)')

    # ── Dataset ───────────────────────────────────────────────────────────────
    log.info('Carregando PneumoniaMNIST...')
    tr_ds   = PneumoniaMNIST(split='train', transform=TRANSFORM, download=True)
    te_ds   = PneumoniaMNIST(split='test',  transform=TRANSFORM, download=True)
    full_ds = ConcatDataset([tr_ds, te_ds])
    labels  = np.array(
        [int(tr_ds[i][1].item()) for i in range(len(tr_ds))] +
        [int(te_ds[i][1].item()) for i in range(len(te_ds))]
    )
    log.info(
        f'Total: {len(full_ds)} amostras | '
        f'Classe 0 (Normal): {np.sum(labels==0)} | '
        f'Classe 1 (Pneumonia): {np.sum(labels==1)}'
    )

    # ── Gerar diagramas dos circuitos VQC ─────────────────────────────────────
    # Limpa arquivo ASCII acumulado de execuções anteriores
    ascii_path = os.path.join(OUT_DIR, 'circuito_vqc_ascii.txt')
    if os.path.exists(ascii_path):
        os.remove(ascii_path)

    log.info('\nGerando diagramas dos circuitos VQC...')
    for nq in [4, 8]:
        log.info(f'  VQC {nq} qubits ({N_LAYERS} camadas)...')
        plot_vqc_circuit(nq, N_LAYERS, OUT_DIR)

    # ── Definir experimentos ──────────────────────────────────────────────────
    idx = np.arange(len(full_ds))
    experiments = [
        # (nome, factory, is_quantum)
        ('CNN-VQC (4 qubits)', lambda: CNN_VQC(models.resnet18(weights=None), 4, N_LAYERS), True),
        ('CNN-VQC (8 qubits)', lambda: CNN_VQC(models.resnet18(weights=None), 8, N_LAYERS), True),
        ('ResNet-18',           get_resnet18,     False),
        ('EfficientNet-B0',     get_efficientnet, False),
        ('CNN Customizada',     get_custom_cnn,   False),
    ]

    # ── Executar todos os experimentos ────────────────────────────────────────
    rows, rocs = [], {}
    for name, factory, is_q in experiments:
        avg, std, rd = run_experiment(
            name, factory, idx, labels, full_ds, is_q, log
        )
        rows.append({
            'Modelo':          name,
            'Acuracia (%)':    f'{avg["accuracy"]} +/- {std["accuracy"]}',
            'Sensibil. (%)':   f'{avg["sensitivity"]} +/- {std["sensitivity"]}',
            'Especif. (%)':    f'{avg["specificity"]} +/- {std["specificity"]}',
            'F1-score':        f'{avg["f1_score"]} +/- {std["f1_score"]}',
            'AUC-ROC':         f'{avg["auc_roc"]} +/- {std["auc_roc"]}',
            'Parametros':      avg['n_params'],
            'Tempo (s)':       avg['train_time_s'],
        })
        rocs[name] = rd

    # ── Salvar Tabela 1 — CSV ─────────────────────────────────────────────────
    df       = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, 'resultados_cenario1.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    log.info(f'\nTabela 1 salva: {csv_path}')
    log.info('\n' + df.to_string(index=False))

    # ── Gerar Figura 4 — Curvas ROC ───────────────────────────────────────────
    plt.figure(figsize=(9, 7))
    cores = ['#1F5C99', '#2E75B6', '#E87722', '#C00000', '#70AD47']
    for (name, _, _), cor in zip(experiments, cores):
        yt, yp      = rocs[name][-1]   # última fold para curva ROC
        fpr, tpr, _ = roc_curve(yt, yp)
        auc         = roc_auc_score(yt, yp)
        plt.plot(fpr, tpr, color=cor, lw=2, label=f'{name} (AUC={auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Classificador aleatório')
    plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=12)
    plt.title(
        'Figura 4 — Curvas ROC: CNN-VQC vs Baselines Clássicos\n'
        'Fonte: Elaborado pelo autor (2026).',
        fontsize=11
    )
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'curvas_roc_cenario1.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f'Figura 4 salva: {fig_path}')
    log.info('Experimento 1 concluido com sucesso.')