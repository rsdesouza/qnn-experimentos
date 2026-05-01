# Redes Neurais Quânticas Aplicadas ao Diagnóstico Assistido de Câncer de Pulmão

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.44.0-purple)](https://pennylane.ai)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0%2Bcu128-orange)](https://pytorch.org)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.4.1-darkblue)](https://www.ibm.com/quantum/qiskit)
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

Este repositório implementa e avalia arquiteturas híbridas **CNN-VQC** (*Convolutional Neural Network* + *Variational Quantum Circuit*) e uma **QCNN** (*Quantum Convolutional Neural Network*) pura para classificação binária de imagens radiológicas torácicas, em complemento e comparação com arquiteturas clássicas baseadas em CNN. O objetivo é investigar a aplicabilidade de **Redes Neurais Quânticas (QNNs)** ao diagnóstico assistido por computador em cenários de Big Data em saúde.

**Questão de pesquisa:**
> Como redes neurais quânticas, em arquiteturas híbridas quântico-clássicas, podem ser aplicadas ao diagnóstico assistido por imagens radiológicas torácicas, e em que medida essas abordagens se comparam a modelos clássicos de Deep Learning em termos de desempenho e viabilidade computacional?

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
│   MEDIÇÃO  ⟨Pauli-Z⟩       │  ⟨Z⟩ ∈ [−1, 1] → P = (1−⟨Z⟩)/2
└─────────────┬──────────────┘
              ↓
     [ Normal / Patológico ]
```

**Frameworks:** PennyLane 0.44 · PyTorch 2.10 · Qiskit Runtime 0.46 (Cenário 4)

---

## Estrutura do Repositório

```
qnn-experimentos/
│
├── experimento1.py                  # Cenário 1 — Comparação direta CNN-VQC vs baselines
├── experimento2.py                  # Cenário 2 — Sensibilidade qubits × camadas
├── experimento3.py                  # Cenário 3 — Escalabilidade por volume de dados
├── experimento4.py                  # Cenário 4 — Inferência em hardware quântico real
├── experimento4_train.py            # Cenário 4 — Treinamento em hardware quântico real
├── experimento5.py                  # Cenário 5 — QCNN pura + generalização poucos dados
│
├── treinar_cnn_vqc_4q_3l.py         # Auxiliar — gera checkpoint na config ótima 4q×3l
├── setup_ibm_quantum.py             # Auxiliar — configura credenciais IBM Quantum
│
├── data/
│   └── README_datasets.md           # Instruções de download e organização dos datasets
│
├── results/                         # ← gerado automaticamente
│   ├── resultados_cenarioN.csv
│   ├── *.png
│   └── circuito_vqc_ascii.txt
│
├── logs/                            # ← gerado automaticamente
│   └── log_experimentoN.txt
│
├── checkpoints/                     # ← gerado automaticamente (modelos por fold)
│
├── requirements.txt                 # Dependências base — Cenários 1, 2, 3 e 5
├── requirements_qiskit.txt          # Dependências adicionais — Cenário 4
└── README.md
```

> **Nota:** As pastas `results/`, `logs/` e `checkpoints/` são criadas automaticamente na primeira execução de qualquer script.

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

```powershell
# 1. Clonar
git clone https://github.com/rsdesouza/qnn-experimentos.git
cd qnn-experimentos

# 2. Criar ambiente virtual
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 3. PyTorch com CUDA 12.8 — instalar ANTES do requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Demais dependências (Cenários 1, 2, 3 e 5)
pip install -r requirements.txt

# 5. Somente para o Cenário 4 (IBM Quantum)
pip install -r requirements_qiskit.txt
```

### Verificar instalação

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
python -c "import pennylane as qml; print('PennyLane:', qml.__version__)"
```

Saída esperada:
```
CUDA: True | NVIDIA GeForce RTX 5060
PennyLane: 0.44.0
```

---

## Datasets

| Dataset | Modalidade | Volume | Cenários | Download |
|---|---|---|---|---|
| **PneumoniaMNIST** | Raio-X | 5.856 imgs | 1, 2, 3, 4 e 5 | Automático via `medmnist` |
| **BreastMNIST** | Ultrassom | 780 imgs | 3 | Automático via `medmnist` |

> Os datasets MedMNIST são utilizados como subrogados acessíveis das tarefas-alvo, com binarização nativa e download automático. A replicação dos experimentos sobre as bases NIH ChestX-ray14, LIDC-IDRI e MIMIC-CXR-JPG está prevista como trabalho futuro (cf. Seção 5.4 da dissertação) e exige credenciais CITI/PhysioNet.

Instruções detalhadas: [`data/README_datasets.md`](data/README_datasets.md)

---

## Experimentos

### Cenário 1 — Comparação Direta

Compara CNN-VQC (4q e 8q) com ResNet-18, EfficientNet-B0 e CNN customizada em validação cruzada 5-fold sobre o PneumoniaMNIST. Avalia acurácia, sensibilidade, especificidade, F1-score e AUC-ROC.

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

### Cenário 2 — Sensibilidade Arquitetural

Análise de sensibilidade em grade 3×3: qubits ∈ {4, 8, 12} × camadas ∈ {1, 2, 3}. Gera heatmaps de AUC-ROC, tempo de treino e contagem de parâmetros para identificar a configuração ótima e caracterizar o fenômeno de barren plateaus.

```bash
python experimento2.py
```

**Achados:** configuração ótima identificada — 4 qubits × 3 camadas — com AUC-ROC = 0,7570 em validação cruzada 3-fold. Tempo total estimado: 8–14 h.

---

### Cenário 3 — Escalabilidade

Avalia tempo de treinamento e inferência do CNN-VQC versus ResNet-18 em função do volume de dados, em duas modalidades: **PneumoniaMNIST** (1k, 5k, 10k, 50k imagens) e **BreastMNIST** (200, 500, 780 imagens), permitindo validação externa por modalidade distinta.

```bash
python experimento3.py
```

**Achados:** overhead computacional do CNN-VQC estável em ~20× sobre a ResNet-18, com crescimento linear do tempo (inclinação log-log = 1,007). Tempo total estimado: ~7 h.

---

### Cenário 4 — Hardware Quântico Real (IBM Quantum)

O Cenário 4 é dividido em **dois experimentos complementares**:

- **`experimento4.py`** — Inferência: aplica o VQC pré-treinado a cem amostras em três condições (simulador exato; hardware sem mitigação; hardware com Zero Noise Extrapolation).
- **`experimento4_train.py`** — Treinamento: conduz uma passada de descida de gradiente diretamente no hardware quântico, com gradientes calculados por parameter-shift rule sobre os 24 parâmetros do VQC.

#### Backend e plano

A IBM Quantum aposentou os processadores **Eagle r3** (`ibm_sherbrooke`, 127 qubits) do plano Open ao longo de 2025–2026 e migrou o acesso gratuito para a família **Heron r2** (156 qubits, topologia heavy-hexagonal, mitigação TLS no chip). O backend específico — entre `ibm_fez`, `ibm_kingston` e `ibm_marrakesh` — é selecionado dinamicamente em tempo de execução pelo método `service.least_busy()` do `qiskit-ibm-runtime`, minimizando tempo de fila.

> **Plano Open:** dez minutos de tempo de QPU por mês, renovados no primeiro dia de cada mês.

#### Configuração inicial — uma única vez por máquina

Etapa **1**: criar a API Key e a Instance no IBM Cloud em https://quantum.cloud.ibm.com/.

Etapa **2**: editar as duas variáveis no início de `setup_ibm_quantum.py`:

```python
TOKEN = "sua_api_key_de_44_caracteres"
CRN   = "crn:v1:bluemix:public:quantum-computing:..."
```

Etapa **3**: executar o helper, que salva as credenciais em `~/.qiskit/qiskit-ibm.json` e valida a conexão:

```powershell
python setup_ibm_quantum.py
```

Após esse passo, os dois experimentos do Cenário 4 leem as credenciais automaticamente.

#### 4.1 — Inferência em hardware

Pré-requisito: ter o checkpoint da configuração ótima 4q×3l (gerado por `treinar_cnn_vqc_4q_3l.py` — script que treina localmente em GPU e inclui sanity-check anti-bug de inversão).

```powershell
python treinar_cnn_vqc_4q_3l.py    # ~10 min na GPU (uma vez)
python experimento4.py             # ~3 min de QPU
```

**Saídas:**
- `results/resultados_cenario4.csv` — Tabela 4 da dissertação
- `results/simulador_vs_hardware_cenario4.png` — Figura 7
- `results/curvas_roc_cenario4.png` — Figura 8

**Achados:** AUC-ROC = 0,8209 (simulador) → 0,8138 (hardware) → 0,8224 (hardware + ZNE). Degradação por ruído marcadamente pequena (0,9% relativo), totalmente recuperada por ZNE. Tempo médio: 2,6 ms (sim.) → 555 ms (hw) → 1.702 ms (hw+ZNE).

#### 4.2 — Treinamento em hardware

```powershell
python experimento4_train.py       # ~5 min de QPU (sem ZNE)
```

Calibrado para o orçamento de 7 minutos: quinze amostras de treino, uma única passada de SGD com taxa de aprendizado 0,20, validação em trinta amostras antes e depois.

**Saídas:**
- `results/resultados_cenario4_train_sem_ZNE.csv` — Tabela 5 da dissertação
- `checkpoints/CNN-VQC_4q_3l_hardware_sem_ZNE.pt`

**Achados:** Δ AUC = -0,0044 após uma iteração de SGD — magnitude inconclusiva sob restrição estatística, estabelecendo um limite superior empírico para o impacto do ruído NISQ sobre uma única atualização de pesos em circuitos rasos.

> **Configurar `USE_ZNE = True`** para treinar com ZNE excede o orçamento mensal (~21 min vs 10 min). Recomenda-se rodar SEM ZNE em um mês e COM ZNE no mês seguinte (renovação dos minutos), totalizando o experimento completo em duas execuções.

#### Sumário de custo de QPU — Cenário 4 completo

| Etapa | Tempo de QPU | Acumulado no mês |
|---|---|---|
| Inferência (3 condições × 100 amostras) | ~3,8 min | 3,8 min |
| Treino sem ZNE (15 amostras × 1 época) | ~5,1 min | 8,9 min |
| Treino com ZNE | ~21 min | indisponível no plano Open |

---

### Cenário 5 — QCNN e Generalização com Poucos Dados

Avalia uma **QCNN pura** (sem backbone clássico de larga escala), com camadas convolucionais e pooling quânticos conforme Cong et al. (2019), testando a capacidade de generalização com conjuntos de treinamento reduzidos. Verifica empiricamente o limite teórico de Caro et al. (2022) sobre QML com poucos dados.

```bash
python experimento5.py
```

Baseado em: [QCNN — PennyLane Glossary](https://pennylane.ai/qml/glossary/qcnn) e [Generalization in QML from few training data](https://pennylane.ai/qml/demos/tutorial_learning_few_data).

---

## Artefatos por Experimento

| Script | Arquivo gerado | Correspondência na dissertação |
|---|---|---|
| `experimento1.py` | `results/resultados_cenario1.csv` | Tabela 1 |
| `experimento1.py` | `results/curvas_roc_cenario1.png` | Figura 4 |
| `experimento1.py` | `results/circuito_vqc_4q.png` | Figura 2a |
| `experimento1.py` | `results/circuito_vqc_8q.png` | Figura 2b |
| `experimento2.py` | `results/resultados_cenario2.csv` | Tabela 2 |
| `experimento2.py` | `results/heatmap_auc_cenario2.png` | Figura 3a |
| `experimento2.py` | `results/heatmap_tempo_cenario2.png` | Figura 3b |
| `experimento2.py` | `results/heatmap_params_cenario2.png` | Figura 3c |
| `experimento2.py` | `results/circuito_qubits_grid.png` | Figura 3d |
| `experimento3.py` | `results/resultados_cenario3.csv` | Tabela 3 |
| `experimento3.py` | `results/escalabilidade_cenario3.png` | Figura 5 |
| `experimento3.py` | `results/escalabilidade_cenario3_breast.png` | Figura 6 |
| `experimento4.py` | `results/resultados_cenario4.csv` | Tabela 4 |
| `experimento4.py` | `results/simulador_vs_hardware_cenario4.png` | Figura 7 |
| `experimento4.py` | `results/curvas_roc_cenario4.png` | Figura 8 |
| `experimento4_train.py` | `results/resultados_cenario4_train_sem_ZNE.csv` | Tabela 5 |
| `experimento5.py` | `results/resultados_cenario5.csv` | Tabela 6 |
| `experimento5.py` | `results/generalizacao_qcnn_cenario5.png` | Figura 9 |

> Cada script gera ainda um log textual em `logs/log_experimentoN.txt` para rastreabilidade.

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
| Python | 3.13.12 |
| PyTorch | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 |
| PennyLane | 0.44.0 |
| pennylane-lightning | 0.44.0 |
| scikit-learn | 1.6.1 |
| NumPy | 2.2.5 |
| pandas | 2.2.3 |
| matplotlib | 3.9.4 |
| seaborn | 0.13.2 |
| medmnist | 3.0.2 |
| pylatexenc | 2.10 |
| **qiskit** | **2.4.1** *(Cenário 4)* |
| **qiskit-ibm-runtime** | **0.46.1** *(Cenário 4)* |
| **qiskit-aer** | **0.17.2** *(Cenário 4)* |

> **Notas sobre dependências do Cenário 4:**
> - `qiskit-transpiler-service` foi removido (deprecado em 2025; serviço cloud restrito a Premium). Usa-se o transpilador local nativo do Qiskit.
> - `mitiq` foi removido (não há versão para Python 3.13). O Zero Noise Extrapolation é feito pela API nativa do `qiskit-ibm-runtime` via `EstimatorV2.options.resilience.zne_mitigation`.
> - `pennylane-qiskit` foi removido (não é importado pelo `experimento4.py`; sua presença forçaria downgrade do `qiskit` para ≤ 2.3.0).

---

## Ambiente de Execução

| Componente | Especificação |
|---|---|
| **SO** | Microsoft Windows 11 Pro (Build 26200) |
| **CPU** | AMD Ryzen 7 5700X · 8 núcleos · 16 threads · 3,4 GHz |
| **GPU** | NVIDIA GeForce RTX 5060 · CUDA 12.8 |
| **RAM** | 32 GB DDR4 |
| **Simulador quântico clássico** | `lightning.qubit` (PennyLane) — C++ |
| **Backend IBM (Cenário 4)** | Família Heron r2 · 156 qubits (`ibm_fez`, `ibm_kingston`, `ibm_marrakesh`) |
| **Plano IBM Quantum** | Open Plan (gratuito) — 10 min de QPU/mês |

**Otimizações implementadas:**
- CNN treinada na GPU com mixed precision quando disponível
- `num_workers=0` no DataLoader (obrigatório Windows/Python 3.13)
- Early stopping + checkpoint por fold para retomada sem reprocessamento
- Submissão em batch agregado de circuitos para o IBM Runtime, minimizando overhead de fila

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
- VERDONE, A. et al. Quantum-enhanced ECG classification with hybrid CNN-VQC. *Quantum Machine Intelligence*, v. 8, n. 12, 2026.

---

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

---

<p align="center">
  Desenvolvido por <strong>Rodolfo da Silva de Souza</strong> · CESAR School · 2026<br>
  Orientação: Prof.ª Dr.ª Pamela Thays Lins Bezerra
</p>