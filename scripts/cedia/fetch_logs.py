import paramiko
import os

HOSTNAME = "hpc.cedia.edu.ec"
USERNAME = "kevin.landazuri__yachaytech.edu.ec"
KEY_FILE = os.path.expanduser(r"~/.ssh/cedia_rsa")
REMOTE_PROJECT_ROOT = "/home/kevin.landazuri__yachaytech.edu.ec/Mateo Gavilanes/S11_PROJECT"

def fetch_logs():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOSTNAME, username=USERNAME, key_filename=KEY_FILE, password="LandaLanda96$")
    
    print("--- s11_master_17086.err ---")
    _, err_tail, _ = ssh.exec_command(f"cat \"{REMOTE_PROJECT_ROOT}/scripts/cedia/s11_master_17086.err\"")
    err_content = err_tail.read().decode().strip()
    print(err_content if err_content else "(Empty error log)")
    
    print("\n--- s11_master_17086.out ---")
    _, out_tail, _ = ssh.exec_command(f"cat \"{REMOTE_PROJECT_ROOT}/scripts/cedia/s11_master_17086.out\"")
    out_content = out_tail.read().decode().strip()
    print(out_content if out_content else "(Empty out log)")
    
    print("\n--- master_experiment_log.json ---")
    _, json_tail, _ = ssh.exec_command(f"ls -l \"{REMOTE_PROJECT_ROOT}/outputs/experiments/master_experiment_log.json\"")
    print(json_tail.read().decode().strip())

    ssh.close()

if __name__ == "__main__":
    fetch_logs()
