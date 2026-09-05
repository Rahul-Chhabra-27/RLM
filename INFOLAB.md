# Running rlmadp on the IITB CSE infolab hosts

Hosts: `ant`, `bee`, `cat`, `dog`, `elk`, `fox` `.cse.iitb.ac.in`.

## The short version

```bash
ssh $USER@login.iitb.ac.in -p 5022      # gateway, if you have no CC LDAP / VPN
ssh bee.cse.iitb.ac.in
exec newgrp infolab                     # BEFORE writing to the shared trees
tmux new -s rlm                         # nothing restarts a dead run here

git clone <this-repo> ~/rlmadp && cd ~/rlmadp
bash setup.sh check                     # audit; changes nothing
bash run_info.sh offline                # works immediately -- no venv, no GPU
```

That last line is not a reduced demo. **The core of rlmadp is stdlib-only**
(`ast`, `io`, `re`, `contextlib`, `dataclasses`, and `urllib` for HTTP), so the
whole RLM runs on a bare login shell with no install, no proxy and no quota
spend. Only serving your own model needs an environment.

## The four rules this project has to obey

**1. `$HOME` is small, NFS-mounted and backed up nightly.** Code and condensed
output only. torch + vLLM unpack to several GB and will hit
`OSError: [Errno 122] Disk quota exceeded`. Worse, a large write to a stalled
filer mount is uninterruptible (`hard,timeo=600` → D-state; Ctrl-C does nothing)
and has taken `$HOME` down host-wide before.

`setup.sh` exports the redirections *before* any install runs, because doing it
afterwards is useless — the wheels are already on the quota:

| Variable | Points at |
|---|---|
| `XDG_CACHE_HOME`, `PIP_CACHE_DIR` | `/mnt/$HOST/data/$USER/cache` |
| `HF_HOME`, `TORCH_HOME` | `/mnt/$HOST/data/$USER/` |
| `VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR` | `/mnt/$HOST/data/$USER/cache` |

**2. Envs go on host-local disk, not NFS.** Different hosts carry different
NVIDIA/CUDA versions, so a venv built on `bee` is not guaranteed to work on
`dog`. `/mnt/$(hostname -s)/data/$USER` keeps them separate by construction.
Archival results go to `/mnt/nas/$USER` (backed up ~monthly).

Override either: `RLMADP_LOCAL`, `RLMADP_ARCHIVE`.

**3. `exec newgrp infolab` first.** A script cannot do this for you — `newgrp`
replaces the shell. Without it your files land with the wrong primary group and
collaborators cannot read them. `setup.sh check` warns if you forgot.

**4. The GPUs are shared and unreserved.** No scheduler, nothing restarts a dead
run, and a card idle at login may be full by the time you serve.
`run_info.sh serve` re-checks free memory immediately before binding, picks the
emptiest card by live utilisation, and sizes `--gpu-memory-utilization` from
`ROOT_NEED_MIB` rather than taking vLLM's default 0.9 — which would claim ~90%
of a shared card and make the host unusable for your co-tenants.

`CUDA_DEVICE_ORDER=PCI_BUS_ID` is exported everywhere: `nvidia-smi` reports PCI
order while CUDA defaults to `FASTEST_FIRST`, so without it the card you
measured as idle is not the card CUDA hands you.

## Internet access

Off by default on the compute hosts. Only needed for `setup.sh serve`.

```bash
bash setup.sh net       # prompts for CC LDAP id + SSO token from sso.iitb.ac.in
```

The token is read with `read -rs` and never enters `argv` or `~/.bash_history`.
This matters: the user-guide's alternative recipe exports your CSE LDAP password
into `http_proxy`, which leaves it in plaintext in your shell history. Access
lapses after ~30 minutes idle — re-run if a download stalls.

## Serving a root model

```bash
bash setup.sh serve                     # venv on local disk, vLLM installed
tmux new -s rlm
GPU=1 bash run_info.sh serve            # window 1
bash run_info.sh run                    # window 2
```

`setup.sh serve` reports the host's driver version before installing, because
the driver — not the wheel index — is what decides which vLLM actually works.
It also appends the host's `/opt/nvidia/hpc_sdk/.../cuda/*/lib64` to
`LD_LIBRARY_PATH` inside the venv's activate script.

A 4B root at `MAXLEN=16384` needs roughly 12 GB: ~8 GB of weights, KV for one
sequence, ~2 GB of activations and overhead. **The root never holds the
document** — it reads REPL observations, ~6k characters a turn — so `MAXLEN` can
stay small. That is the practical payoff of the method on a shared cluster: the
corpus can be 3 MB while the server is sized for a 16k window.

## Commands

| Command | GPU | venv | What |
|---|:-:|:-:|---|
| `setup.sh check` | – | – | environment audit: group, python, quota, GPUs, network, tmux |
| `setup.sh net` | – | – | IITB SSO Internet activation |
| `setup.sh core` | – | – | verify the zero-dependency path |
| `setup.sh serve` | ✓ | builds | vLLM venv on host-local disk |
| `run_info.sh offline` | – | – | full RLM run, scripted root |
| `run_info.sh trace` | – | – | per-function trigger log |
| `run_info.sh guards` | – | – | grounding + abstention guards |
| `run_info.sh serve` | ✓ | ✓ | start the vLLM root server |
| `run_info.sh run` | – | – | RLM against that server (stdlib HTTP client) |
| `run_info.sh sweep` | – | – | chunk-size sweep |

Note `run` needs **no venv**: the client is `urllib`. Only the server side does.
That separation means a broken serve env never blocks a run against a server
someone else already started.

## What the sweep shows

```
chars      answer     coverage     subcalls  end
3000       none       0.29%        3         abstained
6000       none       0.59%        3         abstained
12000      none       1.19%        3         abstained
24000      found      2.39%        3         ok
40000      found      3.99%        3         ok
80000      found      4.00%        3         ok
```

Below ~24,000 characters per `llm_query`, the slice no longer reaches the
planted passage, hop 1 fails, and the chain breaks — the run **abstains** rather
than inventing an answer. Above 40,000 nothing improves, because the scripted
root's windows are 40,000 wide: a cap is a ceiling the root may use, not a floor
that makes it read.

## Housekeeping

- `quota -u $USER` — and note that a **silent** result usually means the cap is
  invisible, not that it is large.
- No `sudo`, no personal anaconda/miniconda. Use the system installs under
  `/opt/anaconda*`, or ask the sysadmin.
- Run vscode from one host only; its state lives in the shared NFS `$HOME` and
  two hosts will clobber it.
- Before you leave: clean local dirs, and leave a README naming what each
  directory you left behind is for.
