import paramiko
import os

HOSTNAME = "hpc.cedia.edu.ec"
USERNAME = "kevin.landazuri__yachaytech.edu.ec"
KEY_FILE = os.path.expanduser(r"~/.ssh/cedia_rsa")
REMOTE_PROJECT_ROOT = "/home/kevin.landazuri__yachaytech.edu.ec/Mateo Gavilanes/S11_PROJECT"

def monitor():
    print("Conectando a CEDIA para monitorear el job 17084...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOSTNAME, username=USERNAME, key_filename=KEY_FILE, password="LandaLanda96$")
    
    # Check queue
    _, out, _ = ssh.exec_command("squeue -j 17084")
    print("\n--- Estado en SLURM (squeue) ---")
    print(out.read().decode().strip())
    
    # Read the end of the log file
    log_file = f"{REMOTE_PROJECT_ROOT}/scripts/cedia/s11_master_17084.out"
    _, out_tail, err_tail = ssh.exec_command(f"tail -n 20 \"{log_file}\"")
    
    log_content = out_tail.read().decode().strip()
    err_content = err_tail.read().decode().strip()
    
    print("\n--- Últimas líneas de ejecución ---")
    if log_content:
        print(log_content)
    elif err_content:
        print(f"Error reading log: {err_content}")
    else:
        print("(El archivo de log aún no tiene contenido o el trabajo está PENDING en la cola).")

    ssh.close()

if __name__ == "__main__":
    monitor()
