"""
=============================================================================
treinar_cnn_vqc_4q_3l.py — Treina o CNN-VQC na configuração ÓTIMA (4×3)
                          [VERSÃO CORRIGIDA — bug de inversão]
=============================================================================
Dissertação:
    Redes Neurais Quânticas Aplicadas ao Diagnóstico Assistido de Câncer
    de Pulmão em Bases Massivas de Imagens Médicas

Autor  : Rodolfo da Silva de Souza
Escola : CESAR School — Mestrado em Engenharia de Software (2026)

CORREÇÃO APLICADA NESTA VERSÃO
------------------------------
A versão anterior gerava um checkpoint cuja AUC ficava INVERTIDA quando
carregada pelo experimento4.py (AUC = 0,055 em vez de algo > 0,5). A causa:
divergência de convenção entre treino e inferência.

  TREINO antigo:    BCEWithLogitsLoss(⟨Z⟩, label)
                    → modelo aprendia ⟨Z⟩ alto = classe 1
  INFERÊNCIA:       P = (1 - ⟨Z⟩) / 2
                    → ⟨Z⟩ alto = classe 0

Como BCEWithLogitsLoss internamente aplica sigmoid(⟨Z⟩), a interpretação
"⟨Z⟩ alto = classe 1" do treino fica oposta à do experimento4.

CORREÇÃO: usar logit = -⟨Z⟩ no cálculo da loss, alinhando o treino
com a convenção da inferência. Não altera nada no experimento4.py.

VALIDAÇÃO ANTI-BUG
------------------
Esta versão acrescenta uma etapa de SANITY-CHECK ao final: roda inferência
sobre 200 amostras de validação usando exatamente a mesma função
expval_to_prob do experimento4 e reporta a AUC. Se ficar < 0,5 (invertida),
o script aborta com erro explícito ao invés de salvar o checkpoint.

Saída
-----
  checkpoints/CNN-VQC_4q_3l_fold0.pt   ← consumido pelo experimento4.py

Tempo estimado
--------------
~5-15 minutos em GPU (RTX 5060 + CUDA 12.8).

Uso
---
  python treinar_cnn_vqc_4q_3l.py
=============================================================================
"""

import os
import sys
import time
import logging
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, accuracy_score

import pennylane as qml
from medmnist import PneumoniaMNIST

warnings.filterwarnings('ignore')

# ── Hiperparâmetros ──────────────────────────────────────────────────────────
SEED       = 42
N_QUBITS   = 4         # ótimo do Cenário 2
N_LAYERS   = 3         # ótimo do Cenário 2 — produz pesos shape (3, 4, 2)
EPOCHS     = 10
BATCH_SIZE = 32
LR         = 1e-3
CKPT_PATH  = os.path.join('checkpoints', 'CNN-VQC_4q_3l_fold0.pt')

# ── GPU / device ─────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE   = torch.device('cuda')
    GPU_NAME = torch.cuda.get_device_name(0)
else:
    DEVICE   = torch.device('cpu')
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
# Definição do CNN-VQC — IDÊNTICA à do experimento4.py
# ══════════════════════════════════════════════════════════════════════════════

def make_vqc_pennylane(n_qubits: int, n_layers: int) -> qml.qnn.TorchLayer:
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

    def forward_expval(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna ⟨Z⟩ ∈ [-1, +1] para cada amostra do batch."""
        feats  = self.cnn(x)
        angles = self.reducer(feats) * np.pi
        outs = []
        for i in range(angles.shape[0]):
            outs.append(self.vqc(angles[i].cpu()))
        return torch.stack(outs).to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_expval(x)


# ══════════════════════════════════════════════════════════════════════════════
# Convenção de inferência (DEVE casar com expval_to_prob do experimento4)
# ══════════════════════════════════════════════════════════════════════════════

def expval_to_prob(expval: float) -> float:
    """Idêntica à do experimento4.py — convenção: ⟨Z⟩=+1 → P=0, ⟨Z⟩=-1 → P=1."""
    p = (1.0 - expval) / 2.0
    return float(np.clip(p, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# Loop de treinamento
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)

    log = logging.getLogger('train_4q_3l')
    log.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter('%(asctime)s  %(message)s'))
    log.addHandler(h)

    log.info('═' * 70)
    log.info('TREINAMENTO — CNN-VQC (4 qubits × 3 camadas)')
    log.info('         [VERSÃO CORRIGIDA — bug de inversão de convenção]')
    log.info('═' * 70)
    log.info(f'Device     : {DEVICE} ({GPU_NAME})')
    log.info(f'Pesos VQC  : ({N_LAYERS}, {N_QUBITS}, 2) = {N_LAYERS * N_QUBITS * 2} parâmetros')
    log.info(f'Épocas     : {EPOCHS}')
    log.info(f'Batch size : {BATCH_SIZE}')
    log.info(f'LR         : {LR}')
    log.info(f'Convenção  : ⟨Z⟩=+1 → P(classe 1)=0   |   ⟨Z⟩=-1 → P(classe 1)=1')
    log.info(f'Logit      : -⟨Z⟩  (alinhado com expval_to_prob do experimento4)')
    log.info(f'Saída      : {CKPT_PATH}')

    # ── Dados ─────────────────────────────────────────────────────────────
    log.info('\nCarregando PneumoniaMNIST...')
    train_ds = PneumoniaMNIST(split='train', transform=TRANSFORM, download=True)
    val_ds   = PneumoniaMNIST(split='val',   transform=TRANSFORM, download=True)
    log.info(f'  Treino: {len(train_ds)} amostras')
    log.info(f'  Val.  : {len(val_ds)} amostras')

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(DEVICE.type == 'cuda')
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(DEVICE.type == 'cuda')
    )

    # ── Modelo ────────────────────────────────────────────────────────────
    model = CNN_VQC(N_QUBITS, N_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    # ── Treinamento ───────────────────────────────────────────────────────
    log.info('\nIniciando treinamento...')
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_epoch = time.time()

        for imgs, lbs in train_loader:
            imgs = imgs.to(DEVICE)
            lbs  = lbs.to(DEVICE).float().view(-1)

            optimizer.zero_grad()
            expval = model.forward_expval(imgs)
            # ─────────────────────────────────────────────────────────────
            # CORREÇÃO DO BUG: o logit é -⟨Z⟩, não ⟨Z⟩.
            # BCEWithLogitsLoss(logit=-⟨Z⟩, y=1)
            # = -log sigmoid(-⟨Z⟩)
            # = -log P(classe 1) onde P=(1-⟨Z⟩)/2 (mod. monotônico)
            # Convenção alinhada com expval_to_prob do experimento4.
            # ─────────────────────────────────────────────────────────────
            logits = -expval
            loss   = criterion(logits, lbs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        dt = time.time() - t_epoch
        log.info(f'  Época {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  tempo={dt:.1f}s')

    t_total = time.time() - t_start
    log.info(f'\nTreinamento concluído em {t_total:.1f}s ({t_total/60:.1f} min)')

    # ══════════════════════════════════════════════════════════════════════
    # SANITY-CHECK ANTI-BUG — calcula AUC sobre validação usando a MESMA
    # função expval_to_prob do experimento4. Se < 0,5 → AUC invertida,
    # checkpoint NÃO é salvo.
    # ══════════════════════════════════════════════════════════════════════
    log.info('\n' + '═' * 70)
    log.info('SANITY-CHECK — verificando alinhamento de convenção')
    log.info('═' * 70)

    model.eval()
    probs_list, labels_list = [], []
    with torch.no_grad():
        for imgs, lbs in val_loader:
            imgs = imgs.to(DEVICE)
            expval = model.forward_expval(imgs).cpu().numpy()
            for ev in expval:
                probs_list.append(expval_to_prob(float(ev)))
            labels_list.extend(lbs.numpy().reshape(-1).tolist())

    val_probs  = np.array(probs_list)
    val_labels = np.array(labels_list)
    val_preds  = (val_probs >= 0.5).astype(int)

    auc = roc_auc_score(val_labels, val_probs)
    acc = accuracy_score(val_labels, val_preds)

    log.info(f'  Validação ({len(val_labels)} amostras):')
    log.info(f'    Acurácia : {acc * 100:.2f}%')
    log.info(f'    AUC-ROC  : {auc:.4f}')
    log.info(f'    Distrib. probs predita: min={val_probs.min():.3f}  '
             f'mediana={np.median(val_probs):.3f}  max={val_probs.max():.3f}')
    log.info(f'    Distrib. labels       : pos={int(val_labels.sum())}  '
             f'neg={int((1-val_labels).sum())}')

    if auc < 0.5:
        log.error('\n✗ ABORT — AUC < 0,5 indica convenção INVERTIDA.')
        log.error('  Investigue antes de salvar o checkpoint.')
        sys.exit(1)

    if auc < 0.6:
        log.warning('\n⚠ Atenção — AUC entre 0,5 e 0,6 sugere modelo pouco efetivo,')
        log.warning('  mas convenção está correta. Salvando o checkpoint mesmo assim.')

    # ── Salvar checkpoint ─────────────────────────────────────────────────
    torch.save(model.state_dict(), CKPT_PATH)
    log.info(f'\n✓ Checkpoint salvo em: {CKPT_PATH}')

    state = torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
    vqc_shape = state['vqc.weights'].shape
    log.info(f'  Pesos VQC: shape={tuple(vqc_shape)} '
             f'(esperado: ({N_LAYERS}, {N_QUBITS}, 2))')

    if tuple(vqc_shape) != (N_LAYERS, N_QUBITS, 2):
        log.error(f'\n✗ Shape divergente!')
        sys.exit(1)

    log.info('\n✓ Tudo OK. Próximo passo:')
    log.info('   python experimento4.py        ← inferência (simulador + hardware)')
    log.info('   python experimento4_train.py  ← treino do VQC no hardware (NOVO)')