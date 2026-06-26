import paramiko
import os
import posixpath

HOSTNAME = "hpc.cedia.edu.ec"
USERNAME = "kevin.landazuri__yachaytech.edu.ec"
KEY_FILE = os.path.expanduser(r"~/.ssh/cedia_rsa")

# Rutas Locales y Remotas
LOCAL_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REMOTE_PROJECT_ROOT = "/home/kevin.landazuri__yachaytech.edu.ec/Mateo Gavilanes/S11_PROJECT"

# Qué subir
FILES_TO_UPLOAD = [
    "scripts/run_full_experiment.py",
    "scripts/cedia/s11_experiment.slurm"
]

DIRS_TO_UPLOAD = [
    "src",
    "data/processed"
]

UPLOAD_SUFFIXES = {".py", ".npz", ".json", ".slurm", ".csv"}

def mkdir_p(sftp, remote_dir):
    parts = [p for p in remote_dir.split("/") if p]
    current = "/" if remote_dir.startswith("/") else "."
    for part in parts:
        current = posixpath.join(current, part)
        try:
            sftp.stat(current)
        except IOError:
            sftp.mkdir(current)

def main():
    print(f"Conectando a {HOSTNAME}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # Use the RSA key, passing the password to decrypt the local private key
    ssh.connect(HOSTNAME, username=USERNAME, key_filename=KEY_FILE, password="LandaLanda96$")
    
    try:
        sftp = ssh.open_sftp()
        try:
            # Subir archivos individuales
            for rel in FILES_TO_UPLOAD:
                local_path = os.path.join(LOCAL_PROJECT_ROOT, rel.replace("/", os.sep))
                remote_path = posixpath.join(REMOTE_PROJECT_ROOT, rel)
                if not os.path.exists(local_path):
                    print(f"SKIP missing {rel}")
                    continue
                mkdir_p(sftp, posixpath.dirname(remote_path))
                print(f"UPLOAD {rel}")
                sftp.put(local_path, remote_path)
            
            # Subir directorios recursivamente
            for rel_dir in DIRS_TO_UPLOAD:
                local_dir = os.path.join(LOCAL_PROJECT_ROOT, rel_dir.replace("/", os.sep))
                if not os.path.isdir(local_dir):
                    continue
                for root, _, files in os.walk(local_dir):
                    for filename in files:
                        if os.path.splitext(filename)[1].lower() not in UPLOAD_SUFFIXES:
                            continue
                        local_path = os.path.join(root, filename)
                        rel = os.path.relpath(local_path, LOCAL_PROJECT_ROOT).replace(os.sep, "/")
                        remote_path = posixpath.join(REMOTE_PROJECT_ROOT, rel)
                        mkdir_p(sftp, posixpath.dirname(remote_path))
                        print(f"UPLOAD {rel}")
                        sftp.put(local_path, remote_path)
        finally:
            sftp.close()
            
        print("Archivos sincronizados en CEDIA. Lanzando SLURM job...")
        
        # Lanzar el job
        cmd = f'cd "{REMOTE_PROJECT_ROOT}/scripts/cedia" && sbatch s11_experiment.slurm'
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print("Resultado SLURM:", stdout.read().decode().strip())
        err = stderr.read().decode().strip()
        if err:
            print("Errores SLURM:", err)
            
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
