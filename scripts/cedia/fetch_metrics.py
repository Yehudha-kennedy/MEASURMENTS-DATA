import paramiko
import os
import json

HOSTNAME = "hpc.cedia.edu.ec"
USERNAME = "kevin.landazuri__yachaytech.edu.ec"
KEY_FILE = os.path.expanduser(r"~/.ssh/cedia_rsa")
REMOTE_PROJECT_ROOT = "/home/kevin.landazuri__yachaytech.edu.ec/Mateo Gavilanes/S11_PROJECT"

def fetch_metrics():
    print("Conectando a CEDIA...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOSTNAME, username=USERNAME, key_filename=KEY_FILE, password="LandaLanda96$")
    
    # Intentar leer el log principal de squeue para ver si acabo
    _, out, _ = ssh.exec_command("squeue -j 17086")
    print(out.read().decode().strip())
    
    # Leer el json resultante
    json_path = f"{REMOTE_PROJECT_ROOT}/outputs/experiments/master_experiment_log.json"
    _, json_out, _ = ssh.exec_command(f"cat \"{json_path}\"")
    
    content = json_out.read().decode('utf-8', errors='replace').strip()
    
    if not content:
        print("\nNo se encontró master_experiment_log.json. Quizas sigue corriendo o falló.")
        # Leer ultimas lineas del out (sin arrojar error de unicode local)
        _, err_tail, _ = ssh.exec_command(f"tail -n 20 \"{REMOTE_PROJECT_ROOT}/scripts/cedia/s11_master_17086.err\"")
        print("\n--- ERRORES (s11_master_17086.err) ---")
        print(err_tail.read().decode('utf-8', errors='replace').strip())
    else:
        try:
            data = json.loads(content)
            print("\n✅ ¡EXPERIMENTO COMPLETADO! Métricas obtenidas:")
            for norm, models in data.items():
                print(f"\n--- Normalización: {norm} ---")
                for model_name, metrics in models.items():
                    # Check if it's a deep model
                    if model_name in ['MLP', 'ResNet-1D', 'KAN']:
                        print(f"[{model_name}] -> F1: {metrics.get('Val_F1', 0):.4f} | Acc: {metrics.get('Val_Acc_mean', 0):.4f} | ROC: {metrics.get('Val_ROC_AUC_mean', 0):.4f}")
        except json.JSONDecodeError:
            print("\nEl archivo no es un JSON válido aún:")
            print(content[:500])

    ssh.close()

if __name__ == "__main__":
    fetch_metrics()
