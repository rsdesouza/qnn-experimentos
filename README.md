# Redes Neurais Quânticas Aplicadas ao Diagnóstico Assistido de Câncer de Pulmão

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.44.1-purple)](https://pennylane.ai)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0%2Bcu128-orange)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/Licença-MIT-lightgrey)](LICENSE)

Repositório oficial dos experimentos da dissertação de mestrado:

> **SOUZA, Rodolfo da Silva de.** Redes neurais quânticas aplicadas ao diagnóstico assistido de câncer de pulmão em bases massivas de imagens médicas. 2026. Dissertação (Mestrado em Engenharia de Software) – CESAR School, Recife, 2026.

**Orientação:** Prof.ª Dr.ª Pamela Thays Lins Bezerra

---

## Sumário

- [Visão Geral](#visão-geral)
- [Arquitetura Proposta](#arquitetura-proposta)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Pré-requisitos e Instalação](#pré-requisitos-e-instalação)
- [Datasets](#datasets)
- [Experimentos](#experimentos)
- [Artefatos por Experimento](#artefatos-por-experimento)
- [Reprodutibilidade](#reprodutibilidade)
- [Ambiente de Execução](#ambiente-de-execução)
- [Citação](#citação)
- [Licença](#licença)

---

## Visão Geral

Este repositório implementa e avalia arquiteturas híbridas **CNN-VQC** (*Convolutional Neural Network* + *Variational Quantum Circuit*) e uma **QCNN** (*Quantum Convolutional Neural Network*) pura para classificação binária de nódulos pulmonares em imagens de tórax. O objetivo é investigar a aplicabilidade de **Redes Neurais Quânticas (QNNs)** ao diagnóstico assistido de câncer de pulmão em cenários de Big Data em saúde.

**Questão de pesquisa:**
> Como redes neurais quânticas, em arquiteturas híbridas quântico-clássicas, podem ser aplicadas ao diagnóstico assistido de câncer de pulmão em bases massivas de imagens médicas, e em que medida essas abordagens se comparam a modelos clássicos de Deep Learning em termos de desempenho e viabilidade computacional?

---

## Arquitetura Proposta

```
[ Imagem de Tórax 224×224 ]
         ↓
┌────────────────────────────┐
│   PRÉ-PROCESSAMENTO        │  Resize · Normalização ImageNet · Grayscale→RGB
└─────────────┬──────────────┘
              ↓
┌────────────────────────────┐
│   CNN  (backbone clássico) │  ResNet-18 → 512 features
│                            │  Reducer: Linear(512→n_qubits) + Tanh × π
└─────────────┬──────────────┘
              ↓   ── FRONTEIRA QUÂNTICA ──
┌────────────────────────────┐
│   ANGLE ENCODING           │  RY(xᵢ)|0⟩  para cada qubit i ∈ [0, n)
└─────────────┬──────────────┘
              ↓
┌────────────────────────────┐
│   VQC / ANSATZ  (L layers) │  RY(θ) + RZ(φ) + CNOT linear
│                            │  Gradientes: parameter-shift rule
└─────────────┬──────────────┘
              ↓
┌────────────────────────────┐
│   MEDIÇÃO  ⟨Pauli-Z⟩       │  ⟨Z⟩ ∈ [−1, 1] → BCEWithLogitsLoss
└─────────────┬──────────────┘
              ↓
     [ Normal / Nódulo ]
```

**Frameworks:** PennyLane 0.44 · PyTorch 2.10 · Qiskit Runtime (Cenário 4)

---

## Estrutura do Repositório

```
qnn_experimentos/
│
├── experimento1.py           # Cenário 1 — Comparação direta CNN-VQC vs baselines
├── experimento2.py           # Cenário 2 — Impacto do número de qubits e camadas
├── experimento3.py           # Cenário 3 — Escalabilidade por volume de dados
├── experimento4.py           # Cenário 4 — Hardware quântico real (IBM Quantum)
├── experimento5.py           # Cenário 5 — QCNN pura + generalização poucos dados
│
├── data/
│   └── README_datasets.md    # Instruções de download e organização dos datasets
│
├── results/                  # ← gerado automaticamente ao rodar os scripts
│   ├── resultados_cenario1.csv
│   ├── resultados_cenario2.csv
│   ├── resultados_cenario3.csv
│   ├── resultados_cenario4.csv
│   ├── resultados_cenario5.csv
│   ├── curvas_roc_cenario1.png
│   ├── circuito_vqc_4q.png
│   ├── circuito_vqc_8q.png
│   ├── circuito_vqc_ascii.txt
│   ├── heatmap_auc_cenario2.png
│   ├── heatmap_tempo_cenario2.png
│   ├── heatmap_params_cenario2.png
│   ├── circuito_qubits_grid.png
│   ├── escalabilidade_cenario3.png
│   ├── simulador_vs_hardware_cenario4.png
│   └── generalizacao_qcnn_cenario5.png
│
├── logs/                     # ← gerado automaticamente ao rodar os scripts
│   ├── log_experimento1.txt
│   ├── log_experimento2.txt
│   ├── log_experimento3.txt
│   ├── log_experimento4.txt
│   └── log_experimento5.txt
│
├── checkpoints/              # ← gerado automaticamente (modelos por fold)
│
├── requirements.txt          # Dependências Python — Cenários 1, 2, 3 e 5
├── requirements_qiskit.txt   # Dependências adicionais — Cenário 4 (IBM Quantum)
├── environment.yml           # Ambiente Conda completo reprodutível
└── README.md
```

> **Nota:** As pastas `results/`, `logs/` e `checkpoints/` são criadas automaticamente na primeira execução de qualquer script. Não é necessário criá-las manualmente.

---

## Pré-requisitos e Instalação

### Requisitos de hardware

| Componente | Mínimo | Testado |
|---|---|---|
| Python | 3.10 | 3.13 |
| CUDA | 11.8 | 12.8 |
| GPU NVIDIA | RTX 3060 | RTX 5060 |
| RAM | 16 GB | 32 GB |
| Disco livre | 20 GB | 50 GB |

### Opção A — pip + venv (recomendado)

```bash
# 1. Clonar
git clone https://github.com/rsdesouza/qnn-experimentos.git
cd qnn-experimentos

# 2. Criar ambiente virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 3. PyTorch com CUDA 12.8 — instalar ANTES do requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Demais dependências (Cenários 1, 2, 3 e 5)
pip install -r requirements.txt

# 5. Somente para o Cenário 4 (IBM Quantum)
pip install -r requirements_qiskit.txt
```

### Opção B — Conda

```bash
conda env create -f environment.yml
conda activate qnn-experimentos

# PyTorch com CUDA 12.8 (pip, pois conda pode não ter cu128)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Verificar instalação

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
python -c "import pennylane as qml; print('PennyLane:', qml.__version__)"
```

Saída esperada:
```
CUDA: True | NVIDIA GeForce RTX 5060
PennyLane: 0.44.1
```

---

## Datasets

| Dataset | Modalidade | Volume | Cenários | Download |
|---|---|---|---|---|
| **PneumoniaMNIST** | Raio-X | 5.856 imgs | 1, 2 e 5 | Automático via `medmnist` |
| **NIH ChestX-ray14** | Raio-X | 112.120 imgs | 1, 2 e 3 | [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) |
| **LIDC-IDRI** | TC 3D | 1.018 TCs | 1 | [TCIA](https://cancerimagingarchive.net/collection/lidc-idri) |
| **MIMIC-CXR-JPG** | Raio-X | 377.110 imgs | 3 | [PhysioNet](https://physionet.org/content/mimic-cxr-jpg) *(credencial CITI)* |

> Todos os scripts utilizam **PneumoniaMNIST por padrão** (download automático, sem cadastro), permitindo execução imediata. Para trocar de dataset, edite apenas o bloco `Dataset` em cada script — a arquitetura CNN-VQC não precisa de alteração.

Instruções detalhadas de download e organização: [`data/README_datasets.md`](data/README_datasets.md)

---

## Experimentos

### Cenário 1 — Comparação Direta

Compara CNN-VQC (4q e 8q) com ResNet-18, EfficientNet-B0 e CNN customizada em validação cruzada 5-fold, avaliando acurácia, sensibilidade, especificidade, F1-score e AUC-ROC.

```bash
python experimento1.py
```

**Tempo estimado (RTX 5060 + Ryzen 7 5700X):**

| Modelo | Por fold | 5 folds |
|---|---|---|
| CNN-VQC (4 qubits) | ~25 min | ~2 h |
| CNN-VQC (8 qubits) | ~55 min | ~4,5 h |
| ResNet-18 | ~4 min | ~20 min |
| EfficientNet-B0 | ~5 min | ~25 min |
| CNN Customizada | ~2 min | ~10 min |

---

### Cenário 2 — Impacto do Número de Qubits

Análise de sensibilidade em grade 3×3: qubits ∈ {4, 8, 12} × camadas ∈ {1, 2, 3}. Gera heatmaps de AUC-ROC, tempo de treino e contagem de parâmetros para identificar a configuração ótima.

```bash
python experimento2.py
```

> Usa imagens 64×64 e 3-fold para manter viável as 9 configurações. Total estimado: 8–14 h.

---

### Cenário 3 — Escalabilidade

Avalia tempo de treinamento e inferência do CNN-VQC versus ResNet-18 em função do volume de dados: {1k, 5k, 10k, 50k imagens} sobre o MIMIC-CXR-JPG.

```bash
python experimento3.py
```

---

### Cenário 4 — Hardware Quântico Real (opcional)

Executa o VQC treinado no Cenário 1 diretamente no IBM Sherbrooke via Qiskit Runtime, comparando desempenho com e sem mitigação de erros por Zero Noise Extrapolation (ZNE).

**Configurar token IBM Quantum (gratuito em https://quantum.ibm.com):**
```bash
python -c "
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel='ibm_quantum', token='SEU_TOKEN_AQUI')
"
```

```bash
python experimento4.py
```

---

### Cenário 5 — QCNN e Generalização com Poucos Dados

Avalia uma QCNN pura (sem backbone clássico) com camadas convolucionais e pooling quânticos, testando a capacidade de generalização com conjuntos de treinamento reduzidos (20 a 2.000 amostras). Verifica empiricamente o limite teórico O(poly(log n)) de Caro et al. (2022).

Baseado em: [Generalization in QML from few training data](https://pennylane.ai/qml/demos/tutorial_learning_few_data) e [QCNN — PennyLane Glossary](https://pennylane.ai/qml/glossary/qcnn).

```bash
python experimento5.py
```

---

## Artefatos por Experimento

Tabela completa de todos os arquivos gerados em `results/` e `logs/`:

| Script | Arquivo gerado | Correspondência na dissertação |
|---|---|---|
| `experimento1.py` | `results/resultados_cenario1.csv` | Tabela 1 |
| `experimento1.py` | `results/curvas_roc_cenario1.png` | Figura 4 |
| `experimento1.py` | `results/circuito_vqc_4q.png` | Figura 2a |
| `experimento1.py` | `results/circuito_vqc_8q.png` | Figura 2b |
| `experimento1.py` | `results/circuito_vqc_ascii.txt` | — (referência interna) |
| `experimento1.py` | `logs/log_experimento1.txt` | — |
| `experimento2.py` | `results/resultados_cenario2.csv` | Tabela 2 |
| `experimento2.py` | `results/heatmap_auc_cenario2.png` | Figura 3a |
| `experimento2.py` | `results/heatmap_tempo_cenario2.png` | Figura 3b |
| `experimento2.py` | `results/heatmap_params_cenario2.png` | Figura 3c |
| `experimento2.py` | `results/circuito_qubits_grid.png` | Figura 3d |
| `experimento2.py` | `logs/log_experimento2.txt` | — |
| `experimento3.py` | `results/resultados_cenario3.csv` | Tabela 3 |
| `experimento3.py` | `results/escalabilidade_cenario3.png` | Figura 5 |
| `experimento3.py` | `logs/log_experimento3.txt` | — |
| `experimento4.py` | `results/resultados_cenario4.csv` | Tabela 4 |
| `experimento4.py` | `results/simulador_vs_hardware_cenario4.png` | Figura 6 |
| `experimento4.py` | `logs/log_experimento4.txt` | — |
| `experimento5.py` | `results/resultados_cenario5.csv` | Tabela 5 |
| `experimento5.py` | `results/generalizacao_qcnn_cenario5.png` | Figura 7 |
| `experimento5.py` | `logs/log_experimento5.txt` | — |

---

## Reprodutibilidade

Todos os scripts utilizam semente global fixa:

```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
```

O sistema de **cache por fold** (`checkpoints/`) permite retomar experimentos interrompidos sem reprocessar folds já concluídos. Para forçar reexecução completa, apague a pasta `checkpoints/`.

### Versões exatas testadas

| Biblioteca | Versão |
|---|---|
| Python | 3.13 |
| PyTorch | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 |
| PennyLane | 0.44.1 |
| pennylane-lightning | 0.44.1 |
| scikit-learn | 1.3.2 |
| NumPy | 1.26.0 |
| pandas | 2.1.0 |
| matplotlib | 3.8.0 |
| seaborn | 0.13.2 |
| medmnist | 3.0.0 |
| pylatexenc | 2.10 |
| qiskit | 1.2.4 *(Cenário 4)* |
| qiskit-ibm-runtime | 0.29.0 *(Cenário 4)* |
| mitiq | 0.38.0 *(Cenário 4)* |
| pennylane-qiskit | 0.39.0 *(Cenário 4)* |

---

## Ambiente de Execução

| Componente | Especificação |
|---|---|
| **SO** | Microsoft Windows 11 Pro (Build 26200) |
| **CPU** | AMD Ryzen 7 5700X · 8 núcleos · 16 threads · 3,4 GHz |
| **GPU** | NVIDIA GeForce RTX 5060 · CUDA 12.8 |
| **RAM** | 32 GB DDR4 |
| **Simulador quântico** | `lightning.qubit` (PennyLane) — C++, alta performance |
| **Backend IBM (Cenário 4)** | `ibm_sherbrooke` · 127 qubits · Eagle r3 |

**Otimizações implementadas:**
- CNN treinada na GPU com AMP float16 (`torch.amp.autocast`)
- VQC paralelizado nos 16 threads do Ryzen via `ThreadPoolExecutor`
- `num_workers=0` no DataLoader (obrigatório Windows/Python 3.13)
- Early stopping + checkpoint por fold para retomada sem reprocessamento

---

## Citação

Se este repositório contribuiu para sua pesquisa, por favor cite:

```bibtex
@mastersthesis{souza2026qnn,
  author  = {Souza, Rodolfo da Silva de},
  title   = {Redes neurais quânticas aplicadas ao diagnóstico assistido
             de câncer de pulmão em bases massivas de imagens médicas},
  school  = {CESAR School},
  year    = {2026},
  address = {Recife, Brasil},
  type    = {Dissertação (Mestrado em Engenharia de Software)}
}
```

**Trabalhos relacionados referenciados neste projeto:**

- CEREZO, M. et al. Variational quantum algorithms. *Nature Reviews Physics*, v. 3, p. 625–644, 2021.
- CARO, M. C. et al. Generalization in quantum machine learning from few training data. *Nature Communications*, v. 13, n. 4919, 2022.
- CONG, I. et al. Quantum convolutional neural networks. *Nature Physics*, v. 15, p. 1273–1278, 2019.
- MCCLEAN, J. R. et al. Barren plateaus in quantum neural network training landscapes. *Nature Communications*, v. 9, n. 4812, 2018.

---

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

---

<p align="center">
  Desenvolvido por <strong>Rodolfo da Silva de Souza</strong> · CESAR School · 2026<br>
  Orientação: Prof.ª Dr.ª Pamela Thays Lins Bezerra
</p>