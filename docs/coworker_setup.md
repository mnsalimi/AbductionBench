# Coworker setup: from a fresh clone to a backed-up benchmark run

Follow these steps in order. Run commands in **Bash** on the computer that will
run the benchmark. The recommended machine is the GPU host, because the model
servers are local to it. macOS also works if you forward both server ports over
SSH. On Windows, use Ubuntu/WSL or an SSH terminal on the GPU host.

The public repository intentionally contains no live credentials. A fresh clone
does not install rclone and does not copy another machine's rclone login.

## 1. Obtain the three private inputs

Ask the project owner privately for:

1. The model-server key. The correctly named variables are `ABENCH_API_KEY` and
   `ABENCH_ADMIN_KEY`. If both servers accept the same key, use it in both.
2. The **complete rclone JSON token**, not just the `access_token` string. It must
   contain `access_token` and `refresh_token`; normally it also has `token_type`
   and `expiry`. A bare access token expires and is not sufficient for a long run.
3. Where the model servers run: either access to the GPU host or current model
   and judge endpoint addresses. Confirm the actual served model names, too.

Do not put these in GitHub, chat logs intended for publication, YAML configs,
`run_log.md`, or source files. The Drive OAuth token grants account-level Drive
access; `root_folder_id` pins this application's destination but does not narrow
the Google OAuth scope. Prefer your own Google login with Editor access to the
shared folder when possible.

## 2. Install programs on the execution machine

Ubuntu/Debian (omit `sudo` if already logged in as root):

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip rclone rsync ripgrep util-linux
bash
```

macOS, with Homebrew already installed:

```bash
brew install python rclone rsync ripgrep
bash
```

On macOS, use the engine's built-in Drive sync. The optional shell sidecar needs
Linux's `flock` and will explain this if unavailable. Do not install or copy a
Linux rclone binary onto macOS: each platform needs its own executable.

Check installation:

```bash
python3 --version
rclone version
rsync --version
```

Python 3.10 or newer is required. The benchmark client itself does not need CUDA
or a GPU; the independently running vLLM servers do.

## 3. Clone the correct version

Use this branch, **not `main` or the older metrics branch**:

```bash
git clone --branch codex/coworker-ready --single-branch https://github.com/mnsalimi/AbductionBench.git
cd AbductionBench
git branch --show-current
git log -1 --oneline
```

The branch command must print `codex/coworker-ready`. If you already have a clone,
save any local changes before switching; do not use `git reset --hard`:

```bash
cd /path/to/AbductionBench
git status
git fetch origin
git switch --track origin/codex/coworker-ready
```

If that local branch already exists, use `git switch codex/coworker-ready` and
`git pull --ff-only origin codex/coworker-ready` instead. An explicit fetch may
be needed in an old single-branch clone:

```bash
git fetch origin codex/coworker-ready:refs/remotes/origin/codex/coworker-ready
```

All commands below run from this repository's root directory.

## 4. Install the Python package

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[adapters,dev]'
abench --help
```

This installs the client, spreadsheet writers, dataset dependencies, and tests.
Some dataset releases install additional author-specific requirements when
prepared. Dataset download/access failures are named in the run report; inspect
them rather than assuming a skipped dataset was evaluated.

## 5. Add model keys and addresses to `.env` locally

```bash
cp .env.coworker.example .env
chmod 600 .env
nano .env
```

Replace both `REPLACE_WITH_SERVER_KEY` placeholders with the real private key.
Do not leave either placeholder unchanged. The relevant lines should be:

```dotenv
ABENCH_API_KEY=<actual private server key>
ABENCH_ADMIN_KEY=<actual private server key>
ABENCH_LOCAL_URL=http://127.0.0.1:18001
ABENCH_JUDGE_URL=http://127.0.0.1:18003
ABENCH_JUDGE_MODEL=gpt-oss-20b-local
HF_TOKEN=
```

The angle-bracketed text is explanatory; replace it rather than copying it
literally. `.env` is ignored by Git. Do not paste the key into model YAML files.

The supplied configs expect **Qwen/Qwen3.5-2B** at port 18001 and
**openai/gpt-oss-20b** at port 18003. They do not start those servers. If different
models are served, ask the maintainer to update the model configurations and
their context-window limits before running. Changing only a URL does not change
the requested model name.

If running on the GPU host, the localhost URLs above are appropriate. If running
on a laptop, open a **second terminal** and keep this SSH connection running:

```bash
ssh -N -L 18001:127.0.0.1:18001 -L 18003:127.0.0.1:18003 user@GPU_HOST
```

Replace `user@GPU_HOST` with the real SSH login. Then retain the localhost URLs in
`.env` on the laptop. Use SSH forwarding or HTTPS rather than sending a server
key over a public unencrypted HTTP connection. The old Cloudflare URLs bundled
in other configs were not reachable during validation; do not assume they work.

Load the variables into the **current terminal**:

```bash
set -a
source .env
set +a
```

Check presence without printing secret values:

```bash
python -c 'import os; print({k: bool(os.environ.get(k)) for k in ("ABENCH_API_KEY", "ABENCH_ADMIN_KEY")})'
```

Both must be `True`. Opening a new terminal requires activating `.venv` and
loading `.env` again. Merely creating `.env` does not automatically load it.

## 6. Import the existing Drive token on THIS execution machine

Having a token on another computer is not enough. It must be imported into the
rclone configuration used by this benchmark process.

For the usual one-line JSON token, use Bash's hidden input prompt:

```bash
read -r -s -p 'Paste the complete rclone JSON token, then press Enter: ' ABENCH_DRIVE_TOKEN_JSON
printf '\n'
printf '%s\n' "$ABENCH_DRIVE_TOKEN_JSON" | bash tools/setup_drive_remote.sh --token-stdin
unset ABENCH_DRIVE_TOKEN_JSON
```

Paste only the `{...}` JSON object, without surrounding shell quotes or rclone's
marker lines. The terminal deliberately will not display what you type. This
avoids a literal secret in shell history. Do not run with shell tracing (`set -x`).

For a multiline JSON token, use a temporary local file instead:

```bash
ABENCH_TOKEN_FILE="$(mktemp)"
chmod 600 "$ABENCH_TOKEN_FILE"
nano "$ABENCH_TOKEN_FILE"
# Paste the complete JSON, save, and close the editor before continuing.
bash tools/setup_drive_remote.sh --token-file "$ABENCH_TOKEN_FILE"
# After successful import, delete only that temporary file:
rm -f -- "$ABENCH_TOKEN_FILE"
unset ABENCH_TOKEN_FILE
```

The script writes a private rclone configuration, normally
`~/.config/rclone/rclone.conf`, and backs up an existing `gdrive` section before
replacing it. `RCLONE_CONFIG`, if set, overrides that location. The script
verifies both **read OK** and **write OK** using a small probe, then removes the
probe. Stop if either check fails.

The configured destination folder is:

https://drive.google.com/drive/folders/1BKmNHYIUIBZsnjeDNfpCbGdJYFl5R9i4

If using a different folder, set `DRIVE_FOLDER_ID` before import. The Google
account represented by the token must have access to it.

If the token lacks `refresh_token`, is revoked, or no longer works, obtain a fresh
authorization instead:

```bash
bash tools/connect_drive.sh
```

Open its authorization link and approve the account with folder access. On a
remote host, forward OAuth port 53682 to your laptop as the script describes.
Do not send a Google password or OTP to the maintainer.

Confirm that this process sees the remote:

```bash
rclone listremotes
rclone lsd gdrive:
```

`gdrive:` must be listed. The new run configs use `remote_path: "gdrive:"` because
the setup already pins the remote to the folder. **Do not add `AbductionBench` a
second time.** Older runs remain under the existing nested `AbductionBench/`
directory; nothing in this version moves or deletes them.

rclone currently warns that its shared Google OAuth client will be retired
during 2026. For durable deployments, create a personal OAuth client using
https://rclone.org/drive/#making-your-own-client-id and reauthorize with that
client. A token generated with a different client must match its client ID and
secret. The import script accepts optional `[client_id] [client_secret]` after
`--token-stdin`; pass private values through variables, not literal history.

## 7. Validate and probe BOTH models

```bash
abench validate configs/runs/coworker_smoke.yaml
abench doctor configs/runs/coworker_smoke.yaml
```

Validation checks configuration/imports. Doctor makes actual discovery/probe
calls, including the batch route. Confirm both expected model names are served
and authentication succeeds. Do not launch the full suite with an unreachable
judge. A supported single-call fallback can work when the batch route is absent,
but is slower.

Common failures:

| Symptom | What to fix |
|---|---|
| `ABENCH_API_KEY ... not set` | Activate the environment and `source .env` with `set -a`. |
| 401 / 403 from model endpoint | Correct the private server key and endpoint. Google tokens do not authenticate vLLM. |
| Connection refused on localhost | Start/check the vLLM servers on the GPU host, or keep both SSH forwards alive. |
| Model not in server's model list | The model-name configuration and actual served model differ. |
| `rclone ... not on PATH` | Install rclone on the machine running the benchmark. |
| `gdrive ... not configured` | Import the JSON token on this machine/user, checking `RCLONE_CONFIG`. |
| Invalid/expired Google token | Reauthorize; a bare access token cannot refresh. |
| Drive write failure / quota / permission error | Check the token's Google account, Editor access, storage quota, and rclone stderr. |

## 8. Run the small complete smoke test FIRST

```bash
abench run configs/runs/coworker_smoke.yaml
```

This uses no dataset downloads. It evaluates two generation and two selection
examples in both IO and COT modes (eight sample outputs total), calls the
reasoning judge only for COT, writes the reports, and uploads to Drive. It can
take several minutes; judge calls cost tokens too.

Copy the exact run ID printed by the command. Check:

```bash
ls runs/<SMOKE_RUN_ID>/reports
rg 'artifact sync|sync_' runs/<SMOKE_RUN_ID>/engine.log runs/<SMOKE_RUN_ID>/events.jsonl
rclone lsf gdrive:<SMOKE_RUN_ID>/reports
```

Replace `<SMOKE_RUN_ID>` with the real directory name; do not type the brackets.
Open `runs/<SMOKE_RUN_ID>/reports/abductionbench_results.xlsx`. In the sample
sheet, COT rows should have `metric.reasoning_*` values; IO rows must have none.
Check `reasoning_metrics_status`, `reasoning_judge_errors`, and
`reasoning_metrics_inapplicable` for missing quantities. Generation differential
elimination and selection branchiness are definition-level exclusions; they
should not be forced to zero. Judge-invalid or undefined ratios are reported.

Also check Drive itself: the new `<SMOKE_RUN_ID>` folder should be **directly
inside the shared target folder**, with the workbook and dataset records.
An evaluation completing locally is not proof of a successful backup. The engine
continues on backup failure, so inspect its backup failure count/log.

## 9. Launch the full benchmark only after the smoke passes

```bash
abench run configs/runs/coworker.yaml
```

This inherits the current full suite: 50 items per dataset where available,
IO/COT tasks, three repeats, and dataset-specific generation/selection modes.
It enables the separate semantic-answer judge **and** COT reasoning judge.
Bulky `raw/` per-batch debug payloads are excluded from backup by default; sample
responses, metrics, checkpoints, logs, and reports are still included. To back up
the raw debug payloads too, explicitly add `-s 'engine.sync.exclude=[]'` to the
run command, allowing for additional Drive requests and storage.
It is a long, expensive run; do not treat the smoke's duration as an estimate.
Check disk space and keep the execution machine alive. Use `tmux`/`screen` on a
remote host if you need the SSH terminal to disconnect.

For a new terminal/session before running or resuming:

```bash
cd /path/to/AbductionBench
source .venv/bin/activate
set -a; source .env; set +a
abench run configs/runs/coworker.yaml --resume <RUN_ID>
```

Resume reuses fingerprint-matching model answers. Missing reasoning metrics can
be added from saved outputs without re-buying inference; the judge may still
need calls. Do not change model names/addresses blindly to resume a different
server and assume it is the same experiment.

New runs upload to `gdrive:<RUN_ID>/`. For an **older nested backup** whose local
run directory is missing, explicitly select its old backup base on restore:

```bash
abench run configs/runs/coworker.yaml --resume <OLD_RUN_ID> -s engine.sync.remote_path=gdrive:AbductionBench
```

The optional Linux sidecar can back up a run that started with sync disabled:

```bash
SYNC_REMOTE='gdrive:' bash tools/sync_run.sh runs/<RUN_ID>
```

Run it in a separate terminal; it loops until stopped. Stop it with Ctrl+C after
the benchmark finishes so its final pass uploads the completed report. **Do not
run it alongside the engine uploader or another sidecar for the same run.**

## 10. What to send the maintainer if anything fails

Send the Git commit (`git rev-parse HEAD`), OS, command used, run ID, doctor
status, relevant `artifact sync` log lines, and the run's `RUN_REPORT.md`.
Never send `.env`, `rclone.conf`, OAuth JSON, raw Authorization headers, or keys.
Do not commit credentials when asking for help.
