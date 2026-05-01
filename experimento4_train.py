"""
=============================================================================
experimento4_train.py — Treinamento do VQC em Hardware Quântico Real
=============================================================================
Dissertação:
    Redes Neurais Quânticas Aplicadas ao Diagnóstico Assistido de Câncer
    de Pulmão em Bases Massivas de Imagens Médicas

Autor  : Rodolfo da Silva de Souza
Escola : CESAR School — Mestrado em Engenharia de Software (2026)
Orient.: Prof.ª Dr.ª Pamela Thays Lins Bezerra

Objetivo
--------
Complementar o experimento4.py (que faz INFERÊNCIA em hardware real)
com um TREINAMENTO efetivamente conduzido no hardware quântico, em
linha com a recomendação metodológica da orientadora:

    "Para entender como os efeitos de erro impactam a performance,
     temos que treinar no hardware também — não apenas testar nele."

A comparação fica:
    - simulador clássico (treinamento já existente do checkpoint)
    - hardware sem ZNE (treinamento conduzido aqui)
    - hardware com ZNE (treinamento conduzido aqui)

Restrições do plano Open IBM Quantum
------------------------------------
O plano Open oferece 10 minutos mensais de QPU. O treinamento por
gradiente em VQC requer parameter-shift rule, que custa 2×P forwards
adicionais por amostra para calcular o gradiente, onde P é o número
de parâmetros do circuito.

    Custo por amostra = 1 (loss) + 2 × 24 (gradiente) = 49 forwards
    Custo por época   = N_AMOSTRAS × 49 forwards

Para cumprir o orçamento de 7 minutos restantes com margem de
segurança, este script foi calibrado para:

    N_TRAIN_SAMPLES  = 15    →  735 forwards de treino (15 × 49)
    N_EPOCHS         = 1     →  uma passada de SGD
    N_VAL_SAMPLES    = 30    →  60 forwards de validação (pré + pós)

Total: 795 circuitos × 0,385 s/circ = ~5,1 min sem ZNE.
Com ZNE (resilience_level=2) o tempo por circuito sobe ~4× para
~1,6 s, totalizando ~21 min — INVIÁVEL no orçamento mensal.
Recomendação: rodar SEM ZNE neste mês; rodar COM ZNE no mês seguinte
quando os 10 minutos forem renovados.

Estratégia metodológica
-----------------------
A CNN clássica (ResNet-18 + reducer) NÃO é treinada no hardware —
isso seria computacionalmente inviável (11M parâmetros vs. 24 do VQC).
Os ângulos de Angle Encoding são PRÉ-COMPUTADOS uma única vez na GPU
local com a CNN congelada do checkpoint válido. Em seguida, treina-se
APENAS os 24 pesos do VQC no hardware.

Esta é a estratégia padrão para QML híbrido em era NISQ — adotada,
por exemplo, em Verdone et al. (2026) e Mitarai et al. (2018).

Saídas geradas
--------------
  results/resultados_cenario4_train.csv             — Tabela 5 da dissertação
  results/curva_treino_hardware_cenario4.png        — Loss/AUC por época
  checkpoints/CNN-VQC_4q_3l_hardware_<cond>.pt      — pesos finais do VQC
  logs/log_experimento4_train.txt                   — Log completo

Configuração do experimento
---------------------------
    USE_ZNE = False  → treinamento sem mitigação (resilience_level=0)
    USE_ZNE = True   → treinamento com ZNE (resilience_level=2)

Recomenda-se rodar duas vezes em meses distintos para comparação
direta sem/com mitigação.
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
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)

import pennylane as qml
from medmnist import PneumoniaMNIST

# ── Imports Qiskit ───────────────────────────────────────────────────────────
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import (
        QiskitRuntimeService,
        EstimatorV2 as Estimator,
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
N_QUBITS   = 4
N_LAYERS   = 3
SHOTS      = 1024

# Orçamento computacional — calibrado para ~5 min em Heron r2
N_TRAIN_SAMPLES = 15      # mini-batch único de treino no hardware
                          # Calibrado para 7 min sem ZNE: 15 × 49 = 735 circs
N_EPOCHS        = 1       # uma passada apenas (orçamento curto)
N_VAL_SAMPLES   = 30      # validação sem gradiente (1 circuito por amostra)
LR              = 0.20    # taxa alta (poucas iterações de SGD)

# Configuração ZNE
ZNE_NOISE_FACTORS = [1.0, 3.0, 5.0]
ZNE_EXTRAPOLATORS = ['linear', 'exponential']

# Selecione a condição que deseja rodar nesta execução
USE_ZNE      = False                       # False = sem mitigação | True = com ZNE
                                           # ATENÇÃO: USE_ZNE=True quadruplica
                                           # o tempo (~0.4s → ~1.6s por circ),
                                           # excedendo os 7 min do plano Open.
                                           # Recomenda-se rodar SEM ZNE primeiro,
                                           # comparar com o checkpoint pré-treino,
                                           # e considerar a versão COM ZNE em
                                           # janela do mês seguinte (10 min novos).
BACKEND_NAME = None                        # None = least_busy

# Caminho do checkpoint inicial (CNN treinada localmente, com correção de bug)
CHECKPOINT_INITIAL = os.path.join('checkpoints', 'CNN-VQC_4q_3l_fold0.pt')

OUT_DIR = 'results'
LOG_DIR = 'logs'

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    GPU_NAME = torch.cuda.get_device_name(0)
else:
    DEVICE = torch.device('cpu')
    GPU_NAME = 'CPU'

torch.manual_seed(SEED)
np.random.seed(SEED)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — CNN-VQC (idêntica aos demais scripts)
# ══════════════════════════════════════════════════════════════════════════════

def make_vqc_pennylane(n_qubits: int, n_layers: int) -> qml.qnn.TorchLayer:
    dev = qml.device('lightning.qubit', wires=n_qubits)

    @qml.qnode(dev, interface='torch', diff_method='parameter-shift')
    def circuit(inputs, weights):
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
        return self.reducer(self.cnn(x)) * np.pi


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — CIRCUITO QISKIT
# ══════════════════════════════════════════════════════════════════════════════

def build_qiskit_circuit(angles: np.ndarray,
                          weights: np.ndarray,
                          n_qubits: int,
                          n_layers: int) -> QuantumCircuit:
    """Idêntico ao circuito do experimento4.py — encoding + ansatz."""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(float(angles[i]), i)
    for layer in range(n_layers):
        for i in range(n_qubits):
            qc.ry(float(weights[layer, i, 0]), i)
            qc.rz(float(weights[layer, i, 1]), i)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
    return qc


def expval_to_prob(expval: float) -> float:
    """Convenção idêntica ao experimento4.py: ⟨Z⟩=+1 → P=0, ⟨Z⟩=-1 → P=1."""
    return float(np.clip((1.0 - expval) / 2.0, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — EXTRAÇÃO DE ÂNGULOS (CNN congelada, executada em GPU local)
# ══════════════════════════════════════════════════════════════════════════════

def precompute_angles(model: CNN_VQC, dataset, indices: np.ndarray,
                       log: logging.Logger) -> tuple[np.ndarray, np.ndarray]:
    """
    Pré-computa os ângulos de Angle Encoding para um conjunto de amostras,
    rodando a CNN clássica (congelada) em GPU local. Apenas esses ângulos
    serão enviados ao hardware quântico; a CNN não é treinada lá.
    """
    model.eval()
    angles_list, labels_list = [], []
    sub = Subset(dataset, indices)
    loader = DataLoader(sub, batch_size=32, shuffle=False, num_workers=0)

    with torch.no_grad():
        for imgs, lbs in loader:
            angs = model.extract_angles(imgs.to(DEVICE)).cpu().numpy()
            angles_list.append(angs)
            labels_list.append(lbs.numpy().reshape(-1))

    angles = np.vstack(angles_list)
    labels = np.concatenate(labels_list)
    log.info(f'  Ângulos pré-computados: shape={angles.shape}')
    return angles, labels


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — EXECUÇÃO DE BATCHES DE CIRCUITOS NO HARDWARE
# ══════════════════════════════════════════════════════════════════════════════

def run_estimator_batch(circuits: list,
                         backend,
                         use_zne: bool,
                         pass_manager,
                         n_qubits: int) -> np.ndarray:
    """
    Executa um lote de circuitos no hardware, retornando array de
    expectation values ⟨Z_0⟩ na mesma ordem.
    """
    isa_circuits  = [pass_manager.run(c) for c in circuits]
    obs_orig      = SparsePauliOp.from_list([('I' * (n_qubits - 1) + 'Z', 1.0)])
    isa_observables = [obs_orig.apply_layout(c.layout) for c in isa_circuits]

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

    pubs = [(c, obs) for c, obs in zip(isa_circuits, isa_observables)]
    job  = estimator.run(pubs)
    result = job.result()
    return np.array([float(r.data.evs) for r in result]), job.job_id()


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 6 — TREINO POR PARAMETER-SHIFT NO HARDWARE
# ══════════════════════════════════════════════════════════════════════════════

def parameter_shift_step(angles_batch: np.ndarray,
                          labels_batch: np.ndarray,
                          weights: np.ndarray,
                          backend, use_zne: bool, pass_manager,
                          n_qubits: int, n_layers: int,
                          log: logging.Logger) -> tuple[np.ndarray, float]:
    """
    Executa UM passo de parameter-shift no hardware:
      1. Constrói (1 + 2P) circuitos por amostra (loss + gradiente)
      2. Empacota TODOS num único job para reduzir overhead de fila
      3. Calcula expectation values
      4. Retorna gradiente médio sobre o batch + loss média

    P = N_LAYERS × N_QUBITS × 2 = 24 parâmetros
    Total de circuitos = N_AMOSTRAS × 49

    Loss usada (BCE com convenção alinhada à inferência):
      logit = -⟨Z⟩
      L     = -y log σ(-⟨Z⟩) - (1-y) log(1 - σ(-⟨Z⟩))
            = -y log P - (1-y) log (1-P)   onde  P = (1-⟨Z⟩)/2 ≈ σ(-⟨Z⟩)

      ∂L/∂⟨Z⟩  = (P - y) ·  (-1)   (em σ(-z))
                 = (y - P)
    Aplicando a regra da cadeia:
      ∂L/∂θ = (y - P) · ∂⟨Z⟩/∂θ
            = (y - P) · ½ [⟨Z⟩(θ+π/2) - ⟨Z⟩(θ-π/2)]
    """
    n_samples = angles_batch.shape[0]
    shape = weights.shape
    n_params = int(np.prod(shape))
    flat_w = weights.reshape(-1).copy()

    # ── Construir lista de TODOS os circuitos ─────────────────────────────
    all_circuits = []
    layout = []   # mapa: (sample_idx, kind, param_idx) — para desempacotar
    SHIFT = np.pi / 2

    log.info(f'    Construindo {n_samples * (1 + 2 * n_params)} circuitos '
             f'({n_samples} amostras × {1 + 2 * n_params} circuitos por amostra)...')

    for s_idx in range(n_samples):
        ang = angles_batch[s_idx]

        # 1) circuito da loss (sem shift)
        all_circuits.append(build_qiskit_circuit(ang, weights, n_qubits, n_layers))
        layout.append((s_idx, 'loss', None))

        # 2) circuitos do gradiente (parameter-shift)
        for p_idx in range(n_params):
            for sign in (+1, -1):
                w_shifted = flat_w.copy()
                w_shifted[p_idx] += sign * SHIFT
                all_circuits.append(build_qiskit_circuit(
                    ang, w_shifted.reshape(shape), n_qubits, n_layers
                ))
                layout.append((s_idx, 'shift', (p_idx, sign)))

    log.info(f'    Submetendo job ao backend ({len(all_circuits)} circuitos)...')
    t0 = time.time()
    expvals, job_id = run_estimator_batch(
        all_circuits, backend, use_zne, pass_manager, n_qubits
    )
    elapsed = time.time() - t0
    log.info(f'    Job {job_id} concluído em {elapsed:.1f}s')

    # ── Desempacotar resultados ───────────────────────────────────────────
    losses = []
    grads_per_sample = np.zeros((n_samples, n_params))

    for i, (s_idx, kind, info) in enumerate(layout):
        if kind == 'loss':
            ev = expvals[i]
            p = (1.0 - ev) / 2.0
            p = np.clip(p, 1e-7, 1 - 1e-7)
            y = labels_batch[s_idx]
            loss_i = -(y * np.log(p) + (1 - y) * np.log(1 - p))
            losses.append(loss_i)
        else:
            p_idx, sign = info
            if sign == +1:
                grads_per_sample[s_idx, p_idx] += 0.5 * expvals[i]
            else:
                grads_per_sample[s_idx, p_idx] -= 0.5 * expvals[i]

    # Gradiente da loss em relação a cada parâmetro:
    #   ∂L/∂θ = (1/N) Σ (y_i - P_i) · ∂⟨Z⟩/∂θ
    grads_mean = np.zeros(n_params)
    for s_idx in range(n_samples):
        ev_s = expvals[layout.index((s_idx, 'loss', None))]
        p_s  = (1.0 - ev_s) / 2.0
        p_s  = np.clip(p_s, 1e-7, 1 - 1e-7)
        y_s  = labels_batch[s_idx]
        coef = (y_s - p_s)   # negativo do coef de ⟨Z⟩, ver dedução acima
        grads_mean += coef * grads_per_sample[s_idx]
    grads_mean /= n_samples

    return grads_mean.reshape(shape), float(np.mean(losses))


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 7 — VALIDAÇÃO POR INFERÊNCIA (sem gradiente, custo barato)
# ══════════════════════════════════════════════════════════════════════════════

def validate_on_hardware(angles_val: np.ndarray, labels_val: np.ndarray,
                          weights: np.ndarray, backend, use_zne: bool,
                          pass_manager, n_qubits: int, n_layers: int,
                          log: logging.Logger) -> dict:
    """Inferência simples (1 circuito por amostra) — para medir AUC."""
    circuits = [
        build_qiskit_circuit(ang, weights, n_qubits, n_layers)
        for ang in angles_val
    ]
    log.info(f'    Inferência de validação ({len(circuits)} circuitos)...')
    t0 = time.time()
    expvals, job_id = run_estimator_batch(
        circuits, backend, use_zne, pass_manager, n_qubits
    )
    elapsed = time.time() - t0

    probs = np.array([expval_to_prob(ev) for ev in expvals])
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels_val, preds).ravel()

    log.info(f'    Validação concluída em {elapsed:.1f}s ({job_id})')
    return {
        'accuracy':    float(accuracy_score(labels_val, preds) * 100),
        'sensitivity': float(tp / (tp + fn + 1e-9) * 100),
        'specificity': float(tn / (tn + fp + 1e-9) * 100),
        'f1_score':    float(f1_score(labels_val, preds)),
        'auc_roc':     float(roc_auc_score(labels_val, probs)),
        'val_time_s':  float(elapsed),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 8 — SELEÇÃO DE BACKEND
# ══════════════════════════════════════════════════════════════════════════════

def select_backend(service, log):
    if BACKEND_NAME is not None:
        log.info(f'  Modo determinístico — backend fixo: {BACKEND_NAME}')
        return service.backend(BACKEND_NAME)
    log.info('  Modo dinâmico — selecionando least_busy() ...')
    return service.least_busy(operational=True, min_num_qubits=100)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    log = logging.getLogger('exp4_train')
    log.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s  %(message)s')
    for h in [
        logging.FileHandler(os.path.join(LOG_DIR, 'log_experimento4_train.txt'),
                            'w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]:
        h.setFormatter(fmt)
        log.addHandler(h)

    log.info('═' * 70)
    log.info('EXPERIMENTO 4 (TREINO) — VQC treinado em hardware quântico real')
    log.info('═' * 70)
    log.info(f'Device clássico  : {DEVICE} ({GPU_NAME})')
    log.info(f'Configuração VQC : {N_QUBITS} qubits × {N_LAYERS} camadas')
    log.info(f'Parâmetros VQC   : {N_LAYERS * N_QUBITS * 2}')
    log.info(f'Amostras treino  : {N_TRAIN_SAMPLES}')
    log.info(f'Épocas           : {N_EPOCHS}')
    log.info(f'Amostras valid.  : {N_VAL_SAMPLES}')
    log.info(f'Learning rate    : {LR}')
    log.info(f'Shots por circ.  : {SHOTS}')
    log.info(f'ZNE habilitado   : {USE_ZNE}')
    log.info(f'Backend          : {BACKEND_NAME or "(least_busy)"}')

    n_params = N_LAYERS * N_QUBITS * 2
    n_circ_train = N_TRAIN_SAMPLES * (1 + 2 * n_params) * N_EPOCHS
    n_circ_val   = N_VAL_SAMPLES
    log.info(f'Circuitos treino : {n_circ_train}')
    log.info(f'Circuitos val.   : {n_circ_val}')
    log.info(f'TOTAL circuitos  : {n_circ_train + n_circ_val}')

    if not QISKIT_OK:
        log.error(f'Falha import Qiskit: {QISKIT_IMPORT_ERROR}')
        sys.exit(1)

    if not os.path.exists(CHECKPOINT_INITIAL):
        log.error(f'Checkpoint inicial não encontrado: {CHECKPOINT_INITIAL}')
        log.error('Execute primeiro: python treinar_cnn_vqc_4q_3l.py')
        sys.exit(1)

    # ── Conectar ao IBM Quantum ───────────────────────────────────────────
    log.info('\nConectando ao IBM Quantum...')
    service = QiskitRuntimeService()
    backend = select_backend(service, log)
    log.info(f'  Backend: {backend.name} | qubits: {backend.num_qubits} | '
             f'fila: {backend.status().pending_jobs}')
    BACKEND_LABEL = backend.name

    # ── Carregar modelo CNN-VQC com checkpoint válido ─────────────────────
    log.info(f'\nCarregando checkpoint inicial: {CHECKPOINT_INITIAL}')
    model = CNN_VQC(N_QUBITS, N_LAYERS).to(DEVICE)
    state = torch.load(CHECKPOINT_INITIAL, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    initial_weights = state['vqc.weights'].cpu().numpy()
    log.info(f'  Pesos iniciais shape: {initial_weights.shape}')

    # ── Preparar dataset ──────────────────────────────────────────────────
    log.info('\nCarregando PneumoniaMNIST...')
    tr_ds = PneumoniaMNIST(split='train', transform=TRANSFORM, download=True)
    te_ds = PneumoniaMNIST(split='test',  transform=TRANSFORM, download=True)
    full_ds = ConcatDataset([tr_ds, te_ds])
    labels_all = np.concatenate([
        np.array([int(tr_ds[i][1].item()) for i in range(len(tr_ds))]),
        np.array([int(te_ds[i][1].item()) for i in range(len(te_ds))])
    ])

    # Subset estratificado: N_TRAIN para treino + N_VAL para validação
    rng = np.random.default_rng(SEED)
    pos_idx = np.where(labels_all == 1)[0]
    neg_idx = np.where(labels_all == 0)[0]
    rng.shuffle(pos_idx); rng.shuffle(neg_idx)

    train_pos = pos_idx[:N_TRAIN_SAMPLES // 2]
    train_neg = neg_idx[:N_TRAIN_SAMPLES // 2]
    val_pos   = pos_idx[N_TRAIN_SAMPLES // 2 : N_TRAIN_SAMPLES // 2 + N_VAL_SAMPLES // 2]
    val_neg   = neg_idx[N_TRAIN_SAMPLES // 2 : N_TRAIN_SAMPLES // 2 + N_VAL_SAMPLES // 2]

    train_idx = np.concatenate([train_pos, train_neg])
    val_idx   = np.concatenate([val_pos,   val_neg])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    log.info(f'  Treino: {len(train_idx)} amostras (balanceado)')
    log.info(f'  Valid.: {len(val_idx)} amostras (balanceado)')

    # ── Pré-computar ângulos (CNN local, congelada) ───────────────────────
    log.info('\nPré-computando ângulos de Angle Encoding (CNN em GPU local)...')
    angles_train, labels_train = precompute_angles(model, full_ds, train_idx, log)
    angles_val,   labels_val   = precompute_angles(model, full_ds, val_idx,   log)

    # ── Pass manager (transpilação) ───────────────────────────────────────
    log.info('\nConfigurando transpilador para o backend...')
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

    # ══════════════════════════════════════════════════════════════════════
    # AVALIAÇÃO INICIAL (antes do treino) — só para ter linha de base
    # ══════════════════════════════════════════════════════════════════════
    log.info('\n' + '═' * 70)
    log.info('ÉPOCA 0 — Avaliação inicial (pré-treino) no hardware')
    log.info('═' * 70)
    metrics_pre = validate_on_hardware(
        angles_val, labels_val, initial_weights,
        backend, USE_ZNE, pm, N_QUBITS, N_LAYERS, log
    )
    log.info(f'  Acc={metrics_pre["accuracy"]:.2f}%  '
             f'AUC={metrics_pre["auc_roc"]:.4f}  '
             f'Sens={metrics_pre["sensitivity"]:.2f}%  '
             f'Esp={metrics_pre["specificity"]:.2f}%')

    # ══════════════════════════════════════════════════════════════════════
    # LOOP DE TREINO
    # ══════════════════════════════════════════════════════════════════════
    weights = initial_weights.copy()
    history = [{'epoch': 0, 'loss': None, **metrics_pre}]

    for epoch in range(1, N_EPOCHS + 1):
        log.info('\n' + '═' * 70)
        log.info(f'ÉPOCA {epoch}/{N_EPOCHS} — passo de gradiente no hardware')
        log.info('═' * 70)
        t_ep = time.time()

        grads, loss = parameter_shift_step(
            angles_train, labels_train, weights,
            backend, USE_ZNE, pm, N_QUBITS, N_LAYERS, log
        )

        # SGD step (descida gradiente puro — Adam exigiria momentum, mais
        # parâmetros pra rastrear, complica em 1 época só)
        weights = weights - LR * grads
        ep_time = time.time() - t_ep

        log.info(f'  Loss média : {loss:.4f}')
        log.info(f'  ‖grad‖     : {np.linalg.norm(grads):.4f}')
        log.info(f'  Tempo época: {ep_time:.1f}s')

        # Validação após o passo
        metrics = validate_on_hardware(
            angles_val, labels_val, weights,
            backend, USE_ZNE, pm, N_QUBITS, N_LAYERS, log
        )
        log.info(f'  Acc={metrics["accuracy"]:.2f}%  '
                 f'AUC={metrics["auc_roc"]:.4f}  '
                 f'Sens={metrics["sensitivity"]:.2f}%  '
                 f'Esp={metrics["specificity"]:.2f}%')
        history.append({'epoch': epoch, 'loss': loss, **metrics})

    # ══════════════════════════════════════════════════════════════════════
    # SALVAR RESULTADOS
    # ══════════════════════════════════════════════════════════════════════
    log.info('\n' + '═' * 70)
    log.info('CONSOLIDANDO RESULTADOS')
    log.info('═' * 70)

    cond_label = 'com_ZNE' if USE_ZNE else 'sem_ZNE'

    # CSV
    df = pd.DataFrame(history)
    df['condicao'] = cond_label
    df['backend']  = BACKEND_LABEL
    csv_path = os.path.join(OUT_DIR, f'resultados_cenario4_train_{cond_label}.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    log.info(f'  CSV: {csv_path}')
    log.info('\n' + df.to_string(index=False))

    # Checkpoint dos pesos finais
    final_state = dict(state)
    final_state['vqc.weights'] = torch.tensor(weights)
    ckpt_path = os.path.join(
        'checkpoints',
        f'CNN-VQC_4q_3l_hardware_{cond_label}.pt'
    )
    torch.save(final_state, ckpt_path)
    log.info(f'  Checkpoint pós-treino: {ckpt_path}')

    # Plot — curva de treino (loss e AUC)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs_x = [h['epoch'] for h in history]

    ax = axes[0]
    losses = [h['loss'] if h['loss'] is not None else np.nan for h in history]
    ax.plot(epochs_x, losses, 'o-', color='#C00000', lw=2, markersize=8)
    ax.set_xlabel('Época', fontsize=11)
    ax.set_ylabel('Loss (BCE)', fontsize=11)
    ax.set_title(f'Loss durante o treino — {BACKEND_LABEL} ({cond_label})')
    ax.grid(alpha=0.3)

    ax = axes[1]
    aucs = [h['auc_roc'] for h in history]
    accs = [h['accuracy'] / 100 for h in history]
    ax.plot(epochs_x, aucs, 'o-', color='#2E75B6', lw=2, markersize=8, label='AUC-ROC')
    ax.plot(epochs_x, accs, 's--', color='#70AD47', lw=2, markersize=8, label='Acurácia')
    ax.axhline(0.5, ls=':', color='gray', alpha=0.6, label='Aleatório')
    ax.set_xlabel('Época', fontsize=11)
    ax.set_ylabel('Métrica', fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Validação a cada época')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        f'Treino do VQC em hardware quântico ({BACKEND_LABEL}) — condição: {cond_label}\n'
        'Fonte: Elaborado pelo autor (2026).',
        fontsize=11
    )
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, f'curva_treino_hardware_{cond_label}.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log.info(f'  Figura: {plot_path}')

    # Análise final
    auc_pre  = history[0]['auc_roc']
    auc_post = history[-1]['auc_roc']
    delta    = auc_post - auc_pre

    log.info('\n' + '═' * 70)
    log.info('ANÁLISE')
    log.info('═' * 70)
    log.info(f'  Backend          : {BACKEND_LABEL}')
    log.info(f'  Condição         : {cond_label}')
    log.info(f'  AUC pré-treino   : {auc_pre:.4f}')
    log.info(f'  AUC pós-treino   : {auc_post:.4f}')
    log.info(f'  Δ AUC            : {delta:+.4f}')
    if delta > 0:
        log.info('  → O treino no hardware MELHOROU o desempenho preditivo.')
    elif delta < 0:
        log.info('  → O treino no hardware PIOROU o desempenho — efeito do ruído '
                 'sobre o gradiente.')
    else:
        log.info('  → Sem mudança detectável.')

    log.info('\nExperimento 4 (treino) concluído com sucesso.')