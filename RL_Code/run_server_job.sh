#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  code/RL_Code/run_server_job.sh <job_name> -- <command> [args...]

Example:
  code/RL_Code/run_server_job.sh ablation_full -- .venv/bin/python code/RL_Code/ablation_experiment.py

The job is detached with nohup + setsid, so it can continue after Cursor is closed.
Logs and PID files are written under server_jobs/<timestamp>_<job_name>/.
USAGE
}

if [[ $# -lt 3 || "${2:-}" != "--" ]]; then
  usage >&2
  exit 2
fi

job_name="$1"
shift 2

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
jobs_root="${SERVER_JOBS_DIR:-${repo_root}/server_jobs}"
stamp="$(date +%Y%m%d_%H%M%S)"
safe_name="$(printf '%s' "${job_name}" | tr -cs 'A-Za-z0-9._-' '_' | sed 's/^_//; s/_$//')"
if [[ -z "${safe_name}" ]]; then
  safe_name="job"
fi

job_dir="${jobs_root}/${stamp}_${safe_name}"
mkdir -p "${job_dir}" "${repo_root}/.mplconfig"

command_file="${job_dir}/command.sh"
log_file="${job_dir}/run.log"
pid_file="${job_dir}/pid"
exit_file="${job_dir}/exit_code"

{
  printf '#!/usr/bin/env bash\n'
  printf 'set +e\n'
  printf 'cd %q\n' "${repo_root}"
  printf 'export PYTHONPATH=%q${PYTHONPATH:+:${PYTHONPATH}}\n' "${script_dir}"
  printf 'export MPLCONFIGDIR=%q\n' "${repo_root}/.mplconfig"
  printf 'echo "--- started_at: $(date -Is) ---"\n'
  printf 'echo "--- cwd: $(pwd) ---"\n'
  printf 'echo "--- command:'
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
  printf ' ---"\n'
  printf 'SECONDS=0\n'
  printf 'set -o pipefail\n'
  printf 'stdbuf -oL -eL'
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
  printf 'code=$?\n'
  printf 'echo "${code}" > %q\n' "${exit_file}"
  printf 'echo "--- exit_code: ${code} elapsed_seconds: ${SECONDS} ended_at: $(date -Is) ---"\n'
  printf 'exit "${code}"\n'
} > "${command_file}"
chmod +x "${command_file}"

nohup setsid bash "${command_file}" > "${log_file}" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" > "${pid_file}"

cat <<INFO
Started server job:
  name: ${job_name}
  pid: ${pid}
  job_dir: ${job_dir}
  log: ${log_file}

Check status:
  code/RL_Code/check_server_jobs.sh

Watch log:
  tail -f ${log_file}
INFO
