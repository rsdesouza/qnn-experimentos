# Redes Neurais Quânticas Aplicadas ao Diagnóstico Assistido de Câncer de Pulmão

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.44%2B-purple)](https://pennylane.ai)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10%2B-orange)](https://pytorch.org)
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
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Datasets](#datasets)
- [Experimentos](#experimentos)
  - [Cenário 1 — Comparação Direta](#cenário-1--comparação-direta)
  - [Cenário 2 — Impacto do Número de Qubits](#cenário-2--impacto-do-número-de-qubits)
  - [Cenário 3 — Escalabilidade](#cenário-3--escalabilidade)
  - [Cenário 4 — Hardware Quântico Real](#cenário-4--hardware-quântico-real-opcional)
- [Resultados](#resultados)
- [Reprodutibilidade](#reprodutibilidade)
- [Ambiente de Execução](#ambiente-de-execução)
- [Citação](#citação)
- [Licença](#licença)

---

## Visão Geral

Este repositório implementa e avalia uma arquitetura híbrida **CNN-VQC** (*Convolutional Neural Network* + *Variational Quantum Circuit*) para classificação binária de nódulos pulmonares em imagens de tórax. O objetivo é investigar a aplicabilidade de **Redes Neurais Quânticas (QNNs)** ao diagnóstico assistido de câncer de pulmão em cenários de Big Data em saúde, comparando o modelo híbrido com redes clássicas de referência.

**Questão de pesquisa central:**
> Como redes neurais quânticas, em arquiteturas híbridas quântico-clássicas, podem ser aplicadas ao diagnóstico assistido de câncer de pulmão em bases massivas de imagens médicas, e em que medida essas abordagens se comparam a modelos clássicos de Deep Learning em termos de desempenho e viabilidade computacional?

---

## Arquitetura Proposta

```
[ Imagem de Tórax ]
        ↓
┌─────────────────────────┐
│   1. PRÉ-PROCESSAMENTO  │  224×224px · Normalização ImageNet · Augmentation
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│   2. CNN (CLÁSSICA)     │  ResNet-18 / EfficientNet-B0
│                         │  → vetor 512/1280 dim → redução para 2ⁿ
└────────────┬────────────┘
             ↓  ── FRONTEIRA QUÂNTICA ──
┌─────────────────────────┐
│   3. ANGLE ENCODING     │  RY(xᵢ)|0⟩  para cada qubit i
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│   4. VQC / ANSATZ       │  RY(θ) + RZ(φ) + CNOT linear · L camadas
│                         │  Gradientes: parameter-shift rule
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│   5. MEDIÇÃO (Pauli-Z)  │  ⟨Z⟩ → sigmoid → P(nódulo)
└────────────┬────────────┘
             ↓
    [ Nódulo / Normal ]
```

**Frameworks:** PennyLane 0.44+ · PyTorch 2.10+ · Qiskit Runtime (Cenário 4)

---

## Estrutura do Repositório

```
qnn_experimentos/
│
├── experimento1.py          # Cenário 1 — Comparação direta CNN-VQC vs baselines
├── experimento2.py          # Cenário 2 — Impacto do número de qubits
├── experimento3.py          # Cenário 3 — Escalabilidade por volume de dados
├── experimento4.py          # Cenário 4 — Hardware quântico real (IBM Quantum)
│
├── data/
│   └── README_datasets.md   # Instruções de download dos datasets
│
├── results/
│   ├── resultados_cenario1.csv
│   ├── resultados_cenario2.csv
│   ├── resultados_cenario3.csv
│   ├── resultados_cenario4.csv
│   ├── curvas_roc_cenario1.png
│   ├── escalabilidade_cenario3.png
│   └── simulador_vs_hardware_cenario4.png
│
├── logs/
│   ├── log_experimento1.txt
│   ├── log_experimento2.txt
│   ├── log_experimento3.txt
│   └── log_experimento4.txt
│
├── requirements.txt         # Dependências Python (simulação)
├── requirements_qiskit.txt  # Dependências adicionais (Cenário 4)
├── environment.yml          # Ambiente Conda completo
└── README.md
```

---

## Pré-requisitos

| Componente | Versão mínima |
|---|---|
| Python | 3.10 |
| CUDA | 12.8 (para GPU NVIDIA) |
| GPU | NVIDIA RTX 5060 ou superior |
| RAM | 16 GB (32 GB recomendado) |
| Disco | 50 GB livres (datasets) |

---

## Instalação

**1. Clonar o repositório:**
```bash
git clone https://github.com/rsdesouza/qnn-experimentos.git
cd qnn-experimentos
```

**2. Criar ambiente virtual:**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

**3. Instalar dependências (com suporte a GPU):**
```bash
# PyTorch com CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Demais dependências
pip install -r requirements.txt
```

**4. Verificar instalação:**
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import pennylane as qml; print('PennyLane:', qml.__version__)"
```

Saída esperada:
```
CUDA: True  NVIDIA GeForce RTX 5060
PennyLane: 0.44.1
```

---

## Datasets

| Dataset | Modalidade | Volume | Uso | Download |
|---|---|---|---|---|
| **PneumoniaMNIST** | Raio-X | 5.856 imgs | Cenários 1 e 2 (automático) | `medmnist` (pip) |
| **NIH ChestX-ray14** | Raio-X | 112.120 imgs | Cenários 1, 2 e 3 | [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) |
| **LIDC-IDRI** | TC 3D | 1.018 TCs | Cenário 1 | [TCIA](https://cancerimagingarchive.net/collection/lidc-idri) |
| **MIMIC-CXR-JPG** | Raio-X | 377.110 imgs | Cenário 3 | [PhysioNet](https://physionet.org/content/mimic-cxr-jpg) *(requer credencial CITI)* |

> Os scripts utilizam **PneumoniaMNIST por padrão** (download automático via `medmnist`), permitindo execução imediata sem cadastros. Para substituir pelo NIH ChestX-ray14 ou LIDC-IDRI, edite o bloco `Dataset` em cada script — a arquitetura CNN-VQC e os demais componentes não precisam de alteração.

---

## Experimentos

### Cenário 1 — Comparação Direta

Avalia acurácia, sensibilidade, especificidade, F1-score e AUC-ROC do CNN-VQC versus baselines clássicos em classificação binária de nódulos pulmonares, com validação cruzada 5-fold.

**Modelos comparados:**
- CNN-VQC (4 qubits)
- CNN-VQC (8 qubits)
- ResNet-18
- EfficientNet-B0
- CNN Customizada (3 blocos convolucionais)

**Executar:**
```bash
python experimento1.py
```

**Saídas geradas:**
```
results/resultados_cenario1.csv   → Tabela 1 da dissertação
results/curvas_roc_cenario1.png   → Figura 4 da dissertação
logs/log_experimento1.txt         → Log completo de treinamento
```

**Tempo estimado (RTX 5060):**
| Modelo | Tempo por fold | Total (5 folds) |
|---|---|---|
| CNN-VQC (4 qubits) | ~25 min | ~2 h |
| CNN-VQC (8 qubits) | ~55 min | ~4,5 h |
| ResNet-18 | ~4 min | ~20 min |
| EfficientNet-B0 | ~5 min | ~25 min |
| CNN Customizada | ~2 min | ~10 min |

---

### Cenário 2 — Impacto do Número de Qubits

Análise de sensibilidade variando qubits em {4, 8, 12} e camadas do ansatz em {1, 2}, avaliando acurácia, AUC-ROC, tempo de treinamento e número de parâmetros.

```bash
python experimento2.py
```

**Saídas:**
```
results/resultados_cenario2.csv   → Tabela 2 da dissertação
```

---

### Cenário 3 — Escalabilidade

Avalia tempo de treinamento e inferência variando o volume de dados em {1.000, 5.000, 10.000, 50.000 imagens}, comparando CNN-VQC e ResNet-18.

```bash
python experimento3.py
```

**Saídas:**
```
results/resultados_cenario3.csv        → Tabela 3 da dissertação
results/escalabilidade_cenario3.png    → Figura 5 da dissertação
```

---

### Cenário 4 — Hardware Quântico Real (opcional)

Executa o VQC no IBM Quantum via Qiskit Runtime, comparando o desempenho em simulador versus hardware real com mitigação de erros por **Zero Noise Extrapolation (ZNE)**.

**Instalar dependências adicionais:**
```bash
pip install -r requirements_qiskit.txt
```

**Configurar credencial IBM Quantum:**
```bash
python -c "
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel='ibm_quantum', token='SEU_TOKEN_AQUI')
"
```
> Token gratuito disponível em: https://quantum.ibm.com

**Executar:**
```bash
python experimento4.py
```

**Saídas:**
```
results/resultados_cenario4.csv                  → Tabela 4 da dissertação
results/simulador_vs_hardware_cenario4.png        → Figura 6 da dissertação
```

---

## Resultados

Os arquivos CSV gerados seguem o formato abaixo e podem ser copiados diretamente para as tabelas da dissertação:

```
Modelo,Acurácia (%),Sensibil. (%),Especif. (%),F1-score,AUC-ROC,Parâmetros,Tempo treino(s)
CNN-VQC (4 qubits),85.3 ± 1.2,83.1 ± 2.1,87.4 ± 1.8,0.8421 ± 0.013,0.9102 ± 0.011,8,4512
...
```

---

## Reprodutibilidade

Todos os experimentos utilizam seed fixa para garantir reprodutibilidade:

```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
```

**Versões exatas utilizadas:**

| Biblioteca | Versão |
|---|---|
| Python | 3.10 |
| PyTorch | 2.10.0+cu128 |
| PennyLane | 0.44.1 |
| torchvision | 0.25.0+cu128 |
| scikit-learn | 1.3.2 |
| NumPy | 1.26.0 |
| pandas | 2.1.0 |
| matplotlib | 3.8.0 |
| medmnist | 3.0.0 |
| qiskit | 1.x (Cenário 4) |
| qiskit-ibm-runtime | 0.20+ (Cenário 4) |

---

## Ambiente de Execução

| Componente | Especificação |
|---|---|
| **SO** | Microsoft Windows 11 Pro (Build 26200) |
| **CPU** | AMD Ryzen 7 5700X · 8 núcleos · 16 threads · 3,4 GHz |
| **GPU** | NVIDIA GeForce RTX 5060 · CUDA 12.8 |
| **RAM** | 32 GB |
| **Simulador quântico** | `lightning.qubit` (PennyLane) |
| **Backend IBM (Cenário 4)** | `ibm_sherbrooke` via Qiskit Runtime |

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

---

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

---

<p align="center">
  Desenvolvido por <strong>Rodolfo da Silva de Souza</strong> · CESAR School · 2026<br>
  Orientação: Prof.ª Dr.ª Pamela Thays Lins Bezerra
</p>