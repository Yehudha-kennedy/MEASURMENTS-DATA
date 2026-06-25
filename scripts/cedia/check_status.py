"""
Check Job Status — CEDIA Monitor
=================================
Checks SLURM queue and tails the latest logs.
"""
import argparse
import sys

from cedia_config import REMOTE_BASE, USERNAME, LOG_PATTERN, ERR_PATTERN
from ssh_utils import force_utf8_stdout, get_ssh_client, run_remote, quote_remote


def main() -> None:
    force_utf8_stdout()
    
    parser = argparse.ArgumentParser(description="Revisa estado de jobs y logs en CEDIA (S11).")
    parser.add_argument("--tail", type=int, default=50, help="Líneas a mostrar de los logs")
    parser.add_argument("--job", help="ID específico del job a consultar")
    args = parser.parse_args()

    ssh = get_ssh_client()
    try:
        print("\n--- 🕒 SQUEUE ---")
        if args.job:
            run_remote(ssh, f"squeue -j {args.job}", check=False)
        else:
            run_remote(ssh, f"squeue -u {USERNAME}", check=False)

        print("\n--- 📁 OUTPUTS ---")
        run_remote(ssh, f"ls -la {quote_remote(REMOTE_BASE)}/outputs 2>/dev/null || echo 'No outputs yet'", check=False)

        if args.job:
            log_name = f"outputs/s11_{args.job}.out"
            err_name = f"outputs/s11_{args.job}.err"
        else:
            # Buscar el más reciente
            print("\nBuscando logs más recientes...")
            _, out, _ = run_remote(ssh, f"cd {quote_remote(REMOTE_BASE)}/outputs && ls -1t *.out 2>/dev/null | head -n 1", check=False)
            recent_log = out.strip()
            log_name = f"outputs/{recent_log}" if recent_log else None
            
            _, out, _ = run_remote(ssh, f"cd {quote_remote(REMOTE_BASE)}/outputs && ls -1t *.err 2>/dev/null | head -n 1", check=False)
            recent_err = out.strip()
            err_name = f"outputs/{recent_err}" if recent_err else None

        if log_name:
            print(f"\n--- 📄 LOG: {log_name} ---")
            run_remote(ssh, f"tail -n {args.tail} {quote_remote(REMOTE_BASE)}/{quote_remote(log_name)} 2>/dev/null || echo 'No existe'", check=False)
            
        if err_name:
            print(f"\n--- ⚠️ ERR: {err_name} ---")
            run_remote(ssh, f"tail -n {args.tail} {quote_remote(REMOTE_BASE)}/{quote_remote(err_name)} 2>/dev/null || echo 'No existe'", check=False)

    finally:
        ssh.close()

if __name__ == "__main__":
    main()
