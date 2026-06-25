"""
Upload & Run — CEDIA Launcher
=============================
Syncs the project source code to CEDIA and optionally submits the SLURM job.
"""
import argparse
import sys
from pathlib import Path

from cedia_config import (
    LOCAL_SRC, LOCAL_CONFIGS, LOCAL_SCRIPTS,
    REMOTE_DIRS, REMOTE_BASE, SLURM_JOB_NAME_PREFIX
)
from ssh_utils import (
    force_utf8_stdout, get_ssh_client, run_remote,
    upload_directory, quote_remote
)


def main() -> None:
    force_utf8_stdout()
    
    parser = argparse.ArgumentParser(description="Sincroniza código a CEDIA y lanza el entrenamiento.")
    parser.add_argument("--skip-sync", action="store_true", help="No subir archivos, solo lanzar")
    parser.add_argument("--no-launch", action="store_true", help="Sincronizar código pero no lanzar job")
    args = parser.parse_args()

    print("🚀 Iniciando pipeline CEDIA para proyecto Mohamed (S11)...")
    
    ssh = get_ssh_client()
    try:
        if not args.skip_sync:
            sftp = ssh.open_sftp()
            try:
                print("\n📦 Sincronizando código fuente (src)...")
                upload_directory(sftp, LOCAL_SRC, REMOTE_DIRS["src"])
                
                print("\n📦 Sincronizando configuraciones (configs)...")
                upload_directory(sftp, LOCAL_CONFIGS, REMOTE_DIRS["configs"])
                
                print("\n📦 Sincronizando scripts (scripts)...")
                upload_directory(sftp, LOCAL_SCRIPTS, REMOTE_DIRS["scripts"], extensions={".py", ".sh"})
                
            finally:
                sftp.close()
            print("✅ Sincronización completada.")

        if not args.no_launch:
            print("\n🚀 Lanzando job en SLURM...")
            
            # Dar permisos de ejecución al script SLURM
            script_path = f"{REMOTE_DIRS['scripts']}/slurm_train.sh"
            run_remote(ssh, f"chmod +x {quote_remote(script_path)}")
            
            # Ejecutar sbatch
            cmd = f"cd {quote_remote(REMOTE_BASE)} && sbatch --parsable {quote_remote(script_path)}"
            code, out, _ = run_remote(ssh, cmd)
            
            if code == 0:
                job_id = out.strip().splitlines()[-1].strip()
                print(f"✅ Job enviado exitosamente! Job ID: {job_id}")
                
                # Mostrar el estado en squeue
                run_remote(ssh, f"squeue -j {job_id}", check=False)
            else:
                print("❌ Error al enviar el job.")

    finally:
        ssh.close()
        print("\nPipeline terminado.")

if __name__ == "__main__":
    main()
