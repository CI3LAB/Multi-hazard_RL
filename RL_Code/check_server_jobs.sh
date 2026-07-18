#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
jobs_root="${SERVER_JOBS_DIR:-${repo_root}/server_jobs}"

if [[ ! -d "${jobs_root}" ]]; then
  echo "No server_jobs directory found: ${jobs_root}"
  exit 0
fi

shopt -s nullglob
job_dirs=("${jobs_root}"/*)
if [[ ${#job_dirs[@]} -eq 0 ]]; then
  echo "No server jobs found under: ${jobs_root}"
  exit 0
fi

printf '%-10s %-8s %-6s %s\n' "STATUS" "PID" "EXIT" "JOB_DIR"
for job_dir in "${job_dirs[@]}"; do
  [[ -d "${job_dir}" ]] || continue
  pid="?"
  if [[ -f "${job_dir}/pid" ]]; then
    pid="$(<"${job_dir}/pid")"
  fi

  exit_code="-"
  if [[ -f "${job_dir}/exit_code" ]]; then
    exit_code="$(<"${job_dir}/exit_code")"
  fi

  status="stopped"
  if [[ "${pid}" != "?" ]] && kill -0 "${pid}" 2>/dev/null; then
    status="running"
  elif [[ "${exit_code}" != "-" ]]; then
    if [[ "${exit_code}" == "0" ]]; then
      status="done"
    else
      status="failed"
    fi
  fi

  printf '%-10s %-8s %-6s %s\n' "${status}" "${pid}" "${exit_code}" "${job_dir}"
done
