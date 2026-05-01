"""
=============================================================================
EXPERIMENTO 4 — VQC em Hardware Quântico Real (IBM Quantum — Heron r2)
=============================================================================
Dissertação:
    Redes Neurais Quânticas Aplicadas ao Diagnóstico Assistido de Câncer
    de Pulmão em Bases Massivas de Imagens Médicas

Autor  : Rodolfo da Silva de Souza
Escola : CESAR School — Mestrado em Engenharia de Software (2026)
Orient.: Prof.ª Dr.ª Pamela Thays Lins Bezerra

Objetivo
--------
Avaliar o impacto do ruído NISQ sobre o desempenho preditivo do VQC
treinado no Cenário 2 (4 qubits × 3 camadas — melhor configuração)
quando executado em hardware quântico real (IBM Quantum — Heron r2).

Compara três condições:
  1. Simulador exato (lightning.qubit) — referência sem ruído
  2. IBM Heron r2 SEM mitigação (resilience_level=0)
  3. IBM Heron r2 COM Zero Noise Extrapolation (resilience_level=2)

Backend alvo
------------
Plano Open (gratuito) atual oferece processadores Heron r2 com 156 qubits
em topologia heavy-hexagonal. Em 2025-2026 a IBM aposentou os Eagle r3
(127 qubits, ex-ibm_sherbrooke) do plano Open. Os processadores Heron r2
são tecnicamente SUPERIORES: até 50× mais rápidos por circuito, suportam
até 5.000 portas de dois qubits por circuito, executam CZ nativamente
(em vez de ECR como o Eagle), e introduzem mitigação TLS no chip.

Escolha do backend específico
-----------------------------
O script seleciona dinamicamente o backend menos ocupado entre os Heron r2
disponíveis na conta (ibm_fez, ibm_kingston, ibm_marrakesh ou similar),
para minimizar tempo de fila no plano Open.

Referência metodológica
-----------------------
Configuração ZNE alinhada com Verdone et al. (2026) — Heron r2 como
backend, mesma família de extrapoladores (linear + exponencial),
noise_factors [1, 3, 5] por digital gate folding.

Procedimento
------------
1. Carrega o estado treinado do CNN-VQC na configuração ótima do Cenário 2
   (4 qubits × 3 camadas), gerado por:
       python treinar_cnn_vqc_4q_3l.py
   Salvo em: checkpoints/CNN-VQC_4q_3l_fold0.pt
2. Extrai apenas os pesos do VQC (24 parâmetros: 3×4×2 = θ,φ por qubit por camada)
3. Reconstrói o circuito equivalente em Qiskit
4. Roda inferência em três condições com um subset de validação reduzido
   (limite prático de execução em hardware real: ~100-200 amostras)
5. Compara métricas: AUC-ROC, acurácia, tempo médio por amostra (s)

Custo de execução em hardware
-----------------------------
Cada amostra requer ~100-1000 shots × 3 noise factors × N transpilações.
Com 100 amostras em Heron r2: ~3-8 minutos de tempo de fila + ~5-15 minutos
de execução (Heron r2 é ~50× mais rápido que Eagle r3).

Saídas geradas
--------------
  results/resultados_cenario4.csv                    — Tabela 4 da dissertação
  results/simulador_vs_hardware_cenario4.png         — Figura da Seção 4.4
  results/curvas_roc_cenario4.png                    — Curvas ROC comparativas
  logs/log_experimento4.txt                          — Log completo

Dependências adicionais
-----------------------
  pip install -r requirements_qiskit.txt

Configurar credenciais IBM Quantum (uma vez):
  python setup_ibm_quantum.py   (preencher TOKEN e CRN no arquivo)

Executar:
  python experimento4.py

ATENÇÃO: este script consome créditos do IBM Quantum. O plano gratuito
(Open Plan) oferece 10 minutos de execução QPU por mês — suficiente para
~50-100 amostras com ZNE.
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
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

import pennylane as qml
from medmnist import PneumoniaMNIST

# ── Imports Qiskit (carregados apenas se hardware estiver disponível) ────────
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import (
        QiskitRuntimeService,
        EstimatorV2 as Estimator,
        Session,
    )
    QISKIT_OK = True
except ImportError as e:
    QISKIT_OK = False
    QISKIT_IMPORT_ERROR = str(e)

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — CONFIGURAÇÃO GLOBAL
# ══════════════════════════════════════════════════════════════════════════════

SEED       = 42
N_QUBITS   = 4         # configuração ótima do Cenário 2
N_LAYERS   = 3         # melhor AUC: 0,7570 com 4 qubits × 3 camadas
N_SAMPLES  = 100       # subset reduzido (limite prático em hardware real)
SHOTS      = 1024      # shots por circuito (precisão da medição)

# Configuração ZNE (alinhada com Verdone et al., 2026)
ZNE_NOISE_FACTORS = [1.0, 3.0, 5.0]
ZNE_EXTRAPOLATORS = ['linear', 'exponential']

# ── Seleção de backend ────────────────────────────────────────────────────────
# Estratégia adotada (a partir de 2026, plano Open IBM Quantum):
#   - Eagle r3 (ex-ibm_sherbrooke, 127 qubits) APOSENTADO do plano Open
#   - Backends disponíveis: Heron r2 com 156 qubits (ibm_fez, ibm_kingston,
#     ibm_marrakesh ou similares — varia por região da conta)
#   - Estratégia: usar service.least_busy() para escolher dinamicamente o
#     menos ocupado entre os Heron r2 acessíveis, minimizando fila
#
# Para forçar um backend específico, defina BACKEND_NAME = 'ibm_xxxx';
# para usar a seleção dinâmica, mantenha BACKEND_NAME = None.
BACKEND_NAME = None    # None = seleção dinâmica via least_busy()

# Filtro mínimo de qubits — Heron r2 tem 156, valor abaixo aceita Eagle/Heron r1
MIN_QUBITS_FILTER = 100

# Caminho do checkpoint — gerado por treinar_cnn_vqc_4q_3l.py com pesos
# shape (N_LAYERS, N_QUBITS, 2). Se você ainda não treinou nesta configuração,
# execute primeiro:
#     python treinar_cnn_vqc_4q_3l.py
CHECKPOINT_C1 = os.path.join(
    'checkpoints',
    f'CNN-VQC_{N_QUBITS}q_{N_LAYERS}l_fold0.pt'
)

OUT_DIR = 'results'
LOG_DIR = 'logs'

# ── GPU / device ──────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    GPU_NAME = torch.cuda.get_device_name(0)
else:
    DEVICE = torch.device('cpu')
    GPU_NAME = 'CPU'

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Transformação ─────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — VQC EM PENNYLANE (REFERÊNCIA SIMULADOR)
# ══════════════════════════════════════════════════════════════════════════════

def make_vqc_pennylane(n_qubits: int, n_layers: int) -> qml.qnn.TorchLayer:
    """
    VQC idêntico ao do Cenário 1 — usado para reproduzir o pipeline
    treinado e extrair as features (ângulos) que serão enviadas a Qiskit.
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
    Arquitetura CNN-VQC do Cenário 1, reconstruída para carregar o
    state_dict salvo. O backbone deve ser idêntico (ResNet-18 sem pesos).
    """

    def __init__(self, n_qubits: int, n_layers: int):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        cnn_out  = backbone.fc.in_features
        self.reducer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_out, n_qubits),
            nn.Tanh()
        )
        self.vqc = make_vqc_pennylane(n_qubits, n_layers)

    def extract_angles(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reproduz o forward até a fronteira clássico→quântica, retornando
        os ângulos que entram no Angle Encoding.
        Este é o output que vamos enviar ao Qiskit.
        """
        return self.reducer(self.cnn(x)) * np.pi   # (batch, n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles  = self.extract_angles(x)
        samples = [angles[i].cpu().detach() for i in range(angles.shape[0])]
        q_out   = [self.vqc(s) for s in samples]
        return torch.stack(q_out).to(x.device).unsqueeze(1)


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — CIRCUITO EQUIVALENTE EM QISKIT
# ══════════════════════════════════════════════════════════════════════════════

def build_qiskit_circuit(
    n_qubits: int,
    n_layers: int,
    angles: np.ndarray,
    weights: np.ndarray
) -> QuantumCircuit:
    """
    Constrói o circuito equivalente em Qiskit, idêntico ao definido
    em PennyLane: Angle Encoding (RY) + ansatz (RY+RZ+CNOT linear).

    Os parâmetros de encoding (angles) e treinados (weights) são
    BIND-eados diretamente no circuito — sem placeholders, pois o
    treinamento já ocorreu.

    Nota: o circuito mede ⟨Z_0⟩ via observável SparsePauliOp na chamada
    do EstimatorV2, não via instrução measure() — o EstimatorV2 trabalha
    com expectation values diretamente. Em hardware Heron r2, o
    transpilador converterá CNOT (cx) para CZ + portas de 1 qubit
    automaticamente, pois CZ é a porta nativa de 2 qubits da família
    Heron (em vez de ECR usada no Eagle r3).
    """
    qc = QuantumCircuit(n_qubits)

    # Estágio 1 — Angle Encoding
    for i in range(n_qubits):
        qc.ry(float(angles[i]), i)

    # Estágio 2 — Ansatz: L camadas de RY+RZ + CNOT linear
    for layer in range(n_layers):
        for i in range(n_qubits):
            qc.ry(float(weights[layer, i, 0]), i)
            qc.rz(float(weights[layer, i, 1]), i)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

    return qc


def expval_to_prob(expval: float) -> float:
    """
    Converte ⟨Z⟩ ∈ [-1, 1] em probabilidade ∈ [0, 1].

    Mapeamento: ⟨Z⟩ = -1 → P=1 (classe positiva certa)
                ⟨Z⟩ = +1 → P=0 (classe negativa certa)

    Equivalente ao sigmoid do logit no pipeline clássico:
      P = (1 - ⟨Z⟩) / 2
    """
    p = (1.0 - expval) / 2.0
    return float(np.clip(p, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — EXTRAÇÃO DOS ÂNGULOS PARA INFERÊNCIA
# ══════════════════════════════════════════════════════════════════════════════

def extract_inference_angles(
    model: CNN_VQC,
    loader: DataLoader,
    n_samples: int,
    log: logging.Logger
) -> tuple[np.ndarray, np.ndarray]:
    """
    Roda a parte clássica (CNN + reducer) e coleta os ângulos de Angle
    Encoding para cada amostra, junto com seus rótulos verdadeiros.

    Estes ângulos serão enviados ao Qiskit para execução do VQC
    em três condições (simulador exato, hardware sem mitigação, com ZNE).
    """
    model.eval()
    angles_list, labels_list = [], []

    with torch.no_grad():
        for imgs, lbs in loader:
            if len(angles_list) >= n_samples:
                break
            angs = model.extract_angles(imgs.to(DEVICE)).cpu().numpy()
            angles_list.append(angs)
            labels_list.append(lbs.numpy().reshape(-1))

    angles = np.vstack(angles_list)[:n_samples]
    labels = np.concatenate(labels_list)[:n_samples]
    log.info(f'  Extraídos: {len(angles)} vetores de ângulos shape={angles.shape}')
    return angles, labels


def extract_vqc_weights(model: CNN_VQC) -> np.ndarray:
    """
    Extrai os pesos treinados do VQC (n_layers, n_qubits, 2)
    do state_dict do modelo carregado.
    """
    for name, param in model.named_parameters():
        if 'vqc' in name and 'weights' in name:
            return param.detach().cpu().numpy()
    raise RuntimeError(
        'Pesos do VQC não encontrados no modelo. '
        'Verifique o nome do parâmetro.'
    )


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — INFERÊNCIA EM TRÊS CONDIÇÕES
# ══════════════════════════════════════════════════════════════════════════════

def run_simulator_exact(
    angles: np.ndarray,
    weights: np.ndarray,
    n_qubits: int,
    n_layers: int,
    log: logging.Logger
) -> tuple[list, float]:
    """
    Condição 1 — Simulador exato (lightning.qubit).
    Referência sem ruído para o cálculo do impacto NISQ.
    """
    log.info(f'  Executando {len(angles)} amostras no simulador exato...')
    vqc = make_vqc_pennylane(n_qubits, n_layers)

    # Carregar pesos treinados no TorchLayer
    with torch.no_grad():
        for name, param in vqc.named_parameters():
            param.copy_(torch.tensor(weights, dtype=torch.float32))

    t0 = time.time()
    probs = []
    for i, ang in enumerate(angles):
        ang_t  = torch.tensor(ang, dtype=torch.float32)
        expval = vqc(ang_t).item()
        probs.append(expval_to_prob(expval))
    elapsed = time.time() - t0
    log.info(f'  Simulador exato: {elapsed:.1f}s ({elapsed/len(angles)*1000:.1f} ms/amostra)')
    return probs, elapsed


def run_hardware(
    angles: np.ndarray,
    weights: np.ndarray,
    n_qubits: int,
    n_layers: int,
    backend,
    use_zne: bool,
    log: logging.Logger
) -> tuple[list, float]:
    """
    Condições 2 e 3 — Execução em hardware quântico real.

    Modo SEM mitigação:
        resilience_level=0 — apenas dynamical decoupling default
    Modo COM ZNE:
        resilience_level=2 — TREX + ZNE (gate folding, fits linear+exp)
    """
    label = 'ZNE' if use_zne else 'sem mitigação'
    log.info(f'  Executando {len(angles)} amostras no {backend.name} '
             f'({label})...')

    # ── 1. Construir circuitos para cada amostra ───────────────────────────
    circuits = [
        build_qiskit_circuit(n_qubits, n_layers, ang, weights)
        for ang in angles
    ]

    # ── 2. Transpilar para o backend (otimização agressiva) ─────────────────
    log.info('    Transpilando circuitos para o ISA do backend...')
    pm = generate_preset_pass_manager(
        optimization_level=3, backend=backend
    )
    isa_circuits = [pm.run(c) for c in circuits]

    # ── 3. Definir observável Pauli-Z no qubit 0 do circuito original ─────
    observable_orig = SparsePauliOp.from_list([('I' * (n_qubits - 1) + 'Z', 1.0)])
    isa_observables = [
        observable_orig.apply_layout(c.layout) for c in isa_circuits
    ]

    # ── 4. Configurar Estimator V2 com ou sem ZNE ───────────────────────────
    estimator = Estimator(mode=backend)
    estimator.options.default_shots = SHOTS

    if use_zne:
        estimator.options.resilience_level = 2
        estimator.options.resilience.zne_mitigation = True
        estimator.options.resilience.zne.noise_factors = ZNE_NOISE_FACTORS
        estimator.options.resilience.zne.extrapolator = ZNE_EXTRAPOLATORS
        estimator.options.resilience.measure_mitigation = True
    else:
        estimator.options.resilience_level = 0

    # ── 5. Executar em PUBs (publish-subscribe units) ──────────────────────
    pubs = [(c, obs) for c, obs in zip(isa_circuits, isa_observables)]

    t0 = time.time()
    job = estimator.run(pubs)
    log.info(f'    Job submetido: {job.job_id()}')
    log.info('    Aguardando execução no backend (pode levar minutos)...')

    result = job.result()
    elapsed = time.time() - t0

    # ── 6. Extrair expectation values e converter para probabilidades ─────
    probs = []
    for pub_result in result:
        expval = float(pub_result.data.evs)
        probs.append(expval_to_prob(expval))

    log.info(f'  {label}: {elapsed:.1f}s ({elapsed/len(angles)*1000:.1f} ms/amostra)')
    return probs, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 6 — MÉTRICAS E VISUALIZAÇÕES
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: list, y_prob: list, thr: float = 0.5) -> dict:
    """Métricas clínicas padrão (idênticas aos demais cenários)."""
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


def plot_simulator_vs_hardware(df: pd.DataFrame, out_dir: str, backend_name: str) -> None:
    """Figura — barras agrupadas comparando AUC, acurácia e tempo nas 3 condições."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cores = ['#1F5C99', '#C00000', '#70AD47']

    ax = axes[0]
    x = np.arange(len(df))
    width = 0.35

    df_plot = df.copy()
    df_plot['AUC-ROC × 100'] = df_plot['AUC-ROC'] * 100

    ax.bar(x - width/2, df_plot['Acuracia (%)'], width,
           label='Acurácia (%)', color='#2E75B6')
    ax.bar(x + width/2, df_plot['AUC-ROC × 100'], width,
           label='AUC-ROC × 100', color='#E87722')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Condicao'], fontsize=9, rotation=15)
    ax.set_ylabel('Métrica (%)', fontsize=11)
    ax.set_title('Desempenho preditivo', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    for i, (acc, auc) in enumerate(zip(df_plot['Acuracia (%)'],
                                       df_plot['AUC-ROC × 100'])):
        ax.text(i - width/2, acc + 1, f'{acc:.1f}', ha='center', fontsize=8)
        ax.text(i + width/2, auc + 1, f'{auc:.1f}', ha='center', fontsize=8)

    ax = axes[1]
    ax.bar(x, df['Tempo medio (ms/amostra)'], color=cores[:len(df)])
    ax.set_xticks(x)
    ax.set_xticklabels(df['Condicao'], fontsize=9, rotation=15)
    ax.set_ylabel('Tempo médio (ms/amostra)', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('Tempo de inferência', fontsize=11)
    ax.grid(alpha=0.3, axis='y')

    for i, t in enumerate(df['Tempo medio (ms/amostra)']):
        ax.text(i, t * 1.1, f'{t:.0f}', ha='center', fontsize=8)

    fig.suptitle(
        f'Figura — Simulador vs. Hardware Quântico Real ({backend_name})\n'
        'Fonte: Elaborado pelo autor (2026).',
        fontsize=11
    )
    plt.tight_layout()
    path = os.path.join(out_dir, 'simulador_vs_hardware_cenario4.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plot] Salvo: {path}')


def plot_roc_curves(results: dict, labels: np.ndarray, out_dir: str, backend_name: str) -> None:
    """Curvas ROC sobrepostas das três condições."""
    fig, ax = plt.subplots(figsize=(8, 7))
    cores = {'Simulador exato': '#1F5C99',
             f'Hardware ({backend_name}) sem ZNE': '#C00000',
             f'Hardware ({backend_name}) com ZNE': '#70AD47'}

    for name, probs in results.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, color=cores.get(name, '#555'), lw=2,
                label=f'{name} (AUC = {auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Classificador aleatório')
    ax.set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=11)
    ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=11)
    ax.set_title(
        f'Curvas ROC: Simulador vs. Hardware Quântico Real ({backend_name})\n'
        'Fonte: Elaborado pelo autor (2026).',
        fontsize=11
    )
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'curvas_roc_cenario4.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plot] Salvo: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 7 — SELEÇÃO DINÂMICA DO BACKEND
# ══════════════════════════════════════════════════════════════════════════════

def select_backend(service: 'QiskitRuntimeService', log: logging.Logger):
    """
    Seleciona o backend a ser utilizado segundo a estratégia configurada:

    - Se BACKEND_NAME estiver definido como string, usa este backend
      especificamente (modo determinístico, útil para reprodutibilidade
      exata).
    - Se BACKEND_NAME for None, usa service.least_busy() para escolher
      dinamicamente o backend operacional menos ocupado dentre aqueles
      acessíveis pela conta com pelo menos MIN_QUBITS_FILTER qubits.
      Esta é a estratégia recomendada no plano Open, pois minimiza o
      tempo de fila — frequentemente reduz de horas para minutos.

    Ambas as estratégias retornam um IBMBackend pronto para uso pelo
    EstimatorV2.
    """
    if BACKEND_NAME is not None:
        log.info(f'  Modo determinístico — usando backend fixo: {BACKEND_NAME}')
        return service.backend(BACKEND_NAME)

    log.info(f'  Modo dinâmico — selecionando backend menos ocupado '
             f'(mín. {MIN_QUBITS_FILTER} qubits)...')
    backend = service.least_busy(
        operational=True,
        min_num_qubits=MIN_QUBITS_FILTER
    )
    return backend


# ══════════════════════════════════════════════════════════════════════════════
# GUARD OBRIGATÓRIO NO WINDOWS (Python 3.13)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # ── Logger ────────────────────────────────────────────────────────────────
    log = logging.getLogger('exp4')
    log.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s  %(message)s')
    for h in [
        logging.FileHandler(os.path.join(LOG_DIR, 'log_experimento4.txt'),
                            'w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]:
        h.setFormatter(fmt)
        log.addHandler(h)

    log.info('═' * 70)
    log.info('EXPERIMENTO 4 — Hardware Quântico Real (IBM Quantum — Heron r2)')
    log.info('═' * 70)
    log.info(f'Device clássico : {DEVICE}  ({GPU_NAME})')
    log.info(f'PyTorch         : {torch.__version__}')
    log.info(f'PennyLane       : {qml.__version__}')
    log.info(f'Python          : {sys.version.split()[0]}')
    log.info(f'Configuração VQC: {N_QUBITS} qubits × {N_LAYERS} camadas')
    log.info(f'Amostras        : {N_SAMPLES} (subset reduzido para hardware)')
    log.info(f'Shots           : {SHOTS}')
    log.info(f'Backend         : {BACKEND_NAME or "(seleção dinâmica least_busy)"}')
    log.info(f'ZNE noise_factors: {ZNE_NOISE_FACTORS}')

    # ── Verificações iniciais ─────────────────────────────────────────────────
    if not QISKIT_OK:
        log.error(f'\nFalha ao importar Qiskit: {QISKIT_IMPORT_ERROR}')
        log.error('Instale com: pip install -r requirements_qiskit.txt')
        sys.exit(1)

    if not os.path.exists(CHECKPOINT_C1):
        log.error(f'\nCheckpoint não encontrado: {CHECKPOINT_C1}')
        log.error(f'Execute primeiro: python treinar_cnn_vqc_4q_3l.py')
        log.error(f'(esse script gera o checkpoint na configuração ótima 4q × 3l)')
        sys.exit(1)

    # ── Conectar ao IBM Quantum ───────────────────────────────────────────────
    log.info('\nConectando ao IBM Quantum...')
    try:
        service = QiskitRuntimeService()
        backend = select_backend(service, log)
        proc_type = backend.processor_type
        family = proc_type.get('family', '?') if isinstance(proc_type, dict) else str(proc_type)
        revision = proc_type.get('revision', '?') if isinstance(proc_type, dict) else ''
        log.info(f'  Backend selecionado: {backend.name}')
        log.info(f'    Qubits           : {backend.num_qubits}')
        log.info(f'    Família          : {family} r{revision}')
        log.info(f'    Status           : {backend.status().status_msg}')
        log.info(f'    Jobs na fila     : {backend.status().pending_jobs}')
    except Exception as e:
        log.error(f'\nFalha ao conectar IBM Quantum: {e}')
        log.error('Configure suas credenciais executando: python setup_ibm_quantum.py')
        sys.exit(1)

    # Nome amigável do backend para títulos das figuras
    BACKEND_LABEL = backend.name

    # ── Carregar dataset e modelo treinado ────────────────────────────────────
    log.info('\nCarregando PneumoniaMNIST...')
    tr_ds   = PneumoniaMNIST(split='train', transform=TRANSFORM, download=True)
    te_ds   = PneumoniaMNIST(split='test',  transform=TRANSFORM, download=True)
    full_ds = ConcatDataset([tr_ds, te_ds])
    labels_all = np.array(
        [int(tr_ds[i][1].item()) for i in range(len(tr_ds))] +
        [int(te_ds[i][1].item()) for i in range(len(te_ds))]
    )

    _, sub_idx = train_test_split(
        np.arange(len(full_ds)),
        test_size=N_SAMPLES / len(full_ds),
        stratify=labels_all,
        random_state=SEED
    )
    val_loader = DataLoader(
        Subset(full_ds, sub_idx),
        batch_size=32, shuffle=False,
        num_workers=0, pin_memory=(DEVICE.type == 'cuda')
    )

    log.info(f'\nCarregando modelo treinado: {CHECKPOINT_C1}')
    model = CNN_VQC(N_QUBITS, N_LAYERS).to(DEVICE)
    state = torch.load(CHECKPOINT_C1, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    log.info('\nExtraindo ângulos de Angle Encoding...')
    angles, labels = extract_inference_angles(model, val_loader, N_SAMPLES, log)
    weights = extract_vqc_weights(model)
    log.info(f'  Pesos VQC shape: {weights.shape} '
             f'(esperado: ({N_LAYERS}, {N_QUBITS}, 2))')

    # ── Executar três condições ───────────────────────────────────────────────
    results = {}
    times = {}

    log.info('\n[1/3] SIMULADOR EXATO (lightning.qubit)')
    probs_sim, t_sim = run_simulator_exact(
        angles, weights, N_QUBITS, N_LAYERS, log
    )
    results['Simulador exato'] = probs_sim
    times['Simulador exato']   = t_sim

    log.info(f'\n[2/3] HARDWARE QUÂNTICO ({BACKEND_LABEL}) — SEM MITIGAÇÃO')
    probs_hw, t_hw = run_hardware(
        angles, weights, N_QUBITS, N_LAYERS,
        backend, use_zne=False, log=log
    )
    name_no_zne = f'Hardware ({BACKEND_LABEL}) sem ZNE'
    results[name_no_zne] = probs_hw
    times[name_no_zne]   = t_hw

    log.info(f'\n[3/3] HARDWARE QUÂNTICO ({BACKEND_LABEL}) — COM ZNE')
    probs_zne, t_zne = run_hardware(
        angles, weights, N_QUBITS, N_LAYERS,
        backend, use_zne=True, log=log
    )
    name_zne = f'Hardware ({BACKEND_LABEL}) com ZNE'
    results[name_zne] = probs_zne
    times[name_zne]   = t_zne

    # ── Compilar Tabela 4 ─────────────────────────────────────────────────────
    log.info('\n' + '═' * 70)
    log.info('CALCULANDO MÉTRICAS')
    log.info('═' * 70)

    rows = []
    for name, probs in results.items():
        m = compute_metrics(labels.tolist(), probs)
        m['Condicao']                   = name
        m['Tempo total (s)']            = round(times[name], 1)
        m['Tempo medio (ms/amostra)']   = round(times[name] / N_SAMPLES * 1000, 2)
        m['Acuracia (%)']               = m.pop('accuracy')
        m['Sensibil. (%)']              = m.pop('sensitivity')
        m['Especif. (%)']               = m.pop('specificity')
        m['F1-score']                   = m.pop('f1_score')
        m['AUC-ROC']                    = m.pop('auc_roc')
        rows.append(m)
        log.info(f'  {name:45} Acc={m["Acuracia (%)"]:5.2f}%  '
                 f'AUC={m["AUC-ROC"]:.4f}  '
                 f'Tempo={m["Tempo medio (ms/amostra)"]:.1f} ms/amostra')

    df = pd.DataFrame(rows)
    df = df[['Condicao', 'Acuracia (%)', 'Sensibil. (%)',
             'Especif. (%)', 'F1-score', 'AUC-ROC',
             'Tempo total (s)', 'Tempo medio (ms/amostra)']]

    csv_path = os.path.join(OUT_DIR, 'resultados_cenario4.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    log.info(f'\nTabela 4 salva: {csv_path}')
    log.info('\n' + df.to_string(index=False))

    # ── Análise comparativa ──────────────────────────────────────────────────
    auc_sim = df.iloc[0]['AUC-ROC']
    auc_hw  = df.iloc[1]['AUC-ROC']
    auc_zne = df.iloc[2]['AUC-ROC']

    deg_sem  = (auc_sim - auc_hw)  / auc_sim * 100 if auc_sim > 0 else 0.0
    rec_zne  = (auc_zne - auc_hw)  / (auc_sim - auc_hw + 1e-9) * 100

    log.info('\n' + '═' * 70)
    log.info('ANÁLISE COMPARATIVA')
    log.info('═' * 70)
    log.info(f'  Backend utilizado                   : {BACKEND_LABEL}')
    log.info(f'  Degradação por ruído (sem mitigação): {deg_sem:.1f}% de AUC perdido')
    log.info(f'  Recuperação por ZNE                 : {rec_zne:.1f}% do gap recuperado')

    log.info('\nGerando figuras...')
    plot_simulator_vs_hardware(df, OUT_DIR, BACKEND_LABEL)
    plot_roc_curves(results, labels, OUT_DIR, BACKEND_LABEL)

    log.info('\nExperimento 4 concluido com sucesso.')