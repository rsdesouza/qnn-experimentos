"""
=============================================================================
setup_ibm_quantum.py — Configuração e validação do acesso ao IBM Quantum
=============================================================================
Configura suas credenciais do IBM Quantum Platform e valida a conexão
ao backend ibm_sherbrooke ANTES de você gastar minutos preciosos do plano
gratuito executando o experimento4.py.

PRÉ-REQUISITOS
--------------
Você precisa de DUAS coisas obtidas em https://quantum.cloud.ibm.com/:

  1. API Key (token):
     - 44 caracteres alfanuméricos
     - Pode ser gerada/copiada na Home dashboard
     - Botão "API key" → "Generate" ou "Copy"

  2. Instance CRN:
     - Cadeia longa começando com "crn:v1:bluemix:public:quantum-computing:..."
     - Visível na seção "Instances" do menu lateral
     - No plano Open vem uma criada por padrão

USO
---
Edite as duas variáveis abaixo (TOKEN e CRN) e execute UMA VEZ:

    python setup_ibm_quantum.py

A configuração é salva em ~/.qiskit/qiskit-ibm.json e dura para sempre,
ou até você rodar este script novamente para sobrescrever.

Após isso, o experimento4.py funcionará sem precisar passar credenciais.
=============================================================================
"""

import sys


# ═══════════════════════════════════════════════════════════════════════════
# 🔑  PREENCHA AS DUAS LINHAS ABAIXO COM SUAS CREDENCIAIS
# ═══════════════════════════════════════════════════════════════════════════

# Cole aqui sua API Key (44 caracteres) — o que você chamou de "API key"
TOKEN = "COLE_AQUI_SUA_API_KEY_DE_44_CARACTERES_ALFANUMÉRICOS"

# Cole aqui o CRN da sua instância (começa com crn:v1:bluemix:...)
# Encontre em https://quantum.cloud.ibm.com/ → Instances
CRN = "COLE_AQUI_O_CRN_DA_SUA_INSTÂNCIA"

# ═══════════════════════════════════════════════════════════════════════════


def main():
    # ── Validação básica das credenciais ──────────────────────────────────
    if "COLE_AQUI" in TOKEN or len(TOKEN) < 30:
        print("❌ ERRO: edite o arquivo e preencha TOKEN com sua API Key.")
        print("   Obtenha em: https://quantum.cloud.ibm.com/")
        sys.exit(1)

    if "..." in CRN or not CRN.startswith("crn:"):
        print("❌ ERRO: edite o arquivo e preencha CRN com o identificador")
        print("   da sua instância. Encontre em:")
        print("   https://quantum.cloud.ibm.com/ → menu lateral → Instances")
        sys.exit(1)

    # ── Importação tardia para mensagem de erro mais clara ────────────────
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        print("❌ ERRO: qiskit-ibm-runtime não instalado.")
        print("   Execute: pip install -r requirements_qiskit.txt")
        sys.exit(1)

    # ── Etapa 1 — Salvar credenciais em ~/.qiskit/qiskit-ibm.json ─────────
    print("─" * 70)
    print("Etapa 1/3: Salvando credenciais em ~/.qiskit/qiskit-ibm.json")
    print("─" * 70)
    try:
        QiskitRuntimeService.save_account(
            channel="ibm_quantum_platform",   # canal atual (substituiu ibm_cloud)
            token=TOKEN,
            instance=CRN,
            set_as_default=True,
            overwrite=True,                    # sobrescrever caso já exista
        )
        print("✓ Credenciais salvas com sucesso.")
    except Exception as e:
        print(f"❌ Falha ao salvar credenciais: {e}")
        sys.exit(1)

    # ── Etapa 2 — Conectar e listar backends acessíveis ───────────────────
    print()
    print("─" * 70)
    print("Etapa 2/3: Conectando ao IBM Quantum e listando backends")
    print("─" * 70)
    try:
        service = QiskitRuntimeService()
        backends = service.backends()
        print(f"✓ Conectado. Backends disponíveis ({len(backends)}):")
        for b in backends:
            try:
                status = b.status().status_msg
                qubits = b.num_qubits
                print(f"    - {b.name:25s} | {qubits:>3} qubits | status: {status}")
            except Exception:
                print(f"    - {b.name}")
    except Exception as e:
        print(f"❌ Falha ao conectar: {e}")
        print()
        print("Causas comuns:")
        print("  - API Key digitada com caracteres faltando/extra")
        print("  - CRN errado (verifique o copy/paste integral)")
        print("  - Sem internet / proxy corporativo bloqueando HTTPS")
        sys.exit(1)

    # ── Etapa 3 — Testar acesso específico ao ibm_sherbrooke ──────────────
    print()
    print("─" * 70)
    print("Etapa 3/3: Validando acesso ao backend ibm_sherbrooke")
    print("─" * 70)
    target = "ibm_sherbrooke"
    try:
        backend = service.backend(target)
        status = backend.status()
        print(f"✓ Backend {target} acessível.")
        print(f"   Qubits         : {backend.num_qubits}")
        print(f"   Status         : {status.status_msg}")
        print(f"   Jobs na fila   : {status.pending_jobs}")
        print(f"   Operacional    : {status.operational}")
    except Exception as e:
        print(f"⚠️  Não foi possível acessar {target}: {e}")
        print()
        print("Possíveis razões:")
        print(f"  - Sua instância não tem permissão para usar {target}")
        print(f"  - Backend temporariamente offline para manutenção")
        print(f"  - Tente um backend alternativo da lista acima editando")
        print(f"    BACKEND_NAME no experimento4.py (ex: 'ibm_brisbane')")
        sys.exit(1)

    # ── Sucesso ───────────────────────────────────────────────────────────
    print()
    print("═" * 70)
    print("✓ CONFIGURAÇÃO CONCLUÍDA COM SUCESSO")
    print("═" * 70)
    print("Próximo passo: execute o experimento")
    print()
    print("    python experimento4.py")
    print()
    print("Aviso: o plano Open oferece 10 minutos de execução por mês.")
    print("Use com parcimônia — o experimento4 foi calibrado para gastar")
    print("aproximadamente 5-7 minutos das 100 amostras × 3 condições.")


if __name__ == "__main__":
    main()