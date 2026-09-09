# Running NIKA with the lab on your Windows machine

`nika` is the one dataset this instance cannot host. Its emulator, Kathará,
needs a container runtime and `CAP_NET_ADMIN`, and neither can be obtained here:
the container's capability **bounding set** (`CapBnd == CapEff ==
0x00000000a80405fb`) excludes `CAP_SYS_ADMIN` and `CAP_NET_ADMIN`, a bounding
set can only shrink, and `unshare -m/-U/-n` all return `EPERM`. Installed
podman to be sure — it fails before it starts with `cannot clone: Operation not
permitted`. Running NIKA's own quickstart here gets as far as
`DockerDaemonConnectionError`.

The release solves this itself. NIKA has a **remote lab-host mode**
(`src/nika/remote/`): a daemon runs where Docker is, the client drives it over
HTTP, and the client is told which port the MCP gateway came up on. So the split
is:

| | where | what runs there |
|---|---|---|
| **lab host** | your Windows 11 machine (WSL2 + Docker Desktop) | Kathará, the emulated network, `nika remote serve` |
| **client** | this instance | the NIKA CLI, AbductionBench, the model and the judge |

Nothing needs a GPU on the lab host — it only runs containers.

## Why the tunnel direction works

Your laptop is behind NAT, so this server cannot dial *it*. But VS Code Remote
SSH already connects laptop → server, and SSH **remote forwarding** (`-R`)
publishes a laptop port on the server's loopback. That is the whole trick: the
server ends up dialling `127.0.0.1`, and SSH carries it back to your machine.
No firewall change, no public exposure, and the MCP gateway — which has no
authentication of its own — never leaves the tunnel.

Two ports are needed, so the gateway port is **pinned** rather than left as the
default `0` (a random free port), which a tunnel cannot follow.

## 1. On Windows: Docker Desktop

Install Docker Desktop and enable the **WSL2 backend** (Settings → General →
*Use the WSL 2 based engine*), then in Settings → Resources → WSL Integration
enable your Ubuntu distribution. Kathará supports Windows through exactly this
path. Check from inside WSL:

```bash
docker run --rm hello-world
```

## 2. On Windows (inside WSL2): NIKA and its images

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # if uv is absent
git clone https://github.com/sands-lab/nika && cd nika
uv sync --extra kathara                                  # the kathara extra is required
```

Pin the gateway port and enable the daemon side. Create `nika.yaml` in the
clone:

```yaml
nika:
  mcp:
    gateway_port: 8701      # pinned: the default 0 picks a random port,
                            # which an SSH tunnel cannot follow
```

Then start the daemon (leave this window running):

```bash
uv run nika remote serve --host 0.0.0.0 --port 8700
```

It prints `daemon listening on http://0.0.0.0:8700`. WSL2 forwards `localhost`
from Windows, so Windows can reach it at `localhost:8700`.

## 3. On Windows: open the tunnel

In a **separate PowerShell window**, alongside your VS Code session:

```powershell
ssh -N -R 8700:localhost:8700 -R 8701:localhost:8701 root@176.182.201.69 -p 9211
```

`-N` means "no shell, just forward". Leave it running. `-R` binds on the
server's *loopback* by default, so nothing is exposed publicly and no `sshd`
change is needed.

Confirm from this instance:

```bash
curl -s http://127.0.0.1:8700/health
```

## 4. On this instance: point NIKA at the lab, deploy, evaluate

```bash
cd /tmp/nika          # or wherever the client clone lives
cat > nika.yaml <<'EOF'
nika:
  remote:
    enabled: true
    url: http://127.0.0.1:8700    # the tunnel's server-side end
  mcp:
    gateway_port: 8701
EOF

nika env list
nika env run dc_clos --size s     # deploys on your laptop, via the daemon
```

The client is handed the gateway port; with it pinned to 8701 the gateway is at
`http://127.0.0.1:8701` on this side of the tunnel. Give AbductionBench those
two facts and run the dataset:

```bash
export ABENCH_NIKA_GATEWAY_URL=http://127.0.0.1:8701
export ABENCH_NIKA_SESSION_ID=<the session id nika env run printed>
abench run configs/runs/full.yaml -d nika
```

The adapter needs no change: it drives the session's published MCP tools
(`ping_pair`, `traceroute`, `frr_show_ip_route`, packet capture), then
`begin_submission_mcp_phase` and `submit`, and scores `rca_f1` against the
ground truth that ships in the repository.

## What is verified and what is not

**Verified here:** the adapter's case loading and scoring against release 0.2.0
(85 incidents — a correct submission scores 1.0, a wrong one 0.0, a partial one
0.667); that `nika env list` works on this instance; that the only thing missing
is the Docker daemon; and that the release's remote mode exists with the ports
above.

**Not verified:** the tunnel and remote deploy end to end, because there is no
Docker host to point at yet. Expect to adjust the port numbers if 8700/8701 are
taken on either side.

## The cheaper thing to check first

Kathará runs one container per network device, so a small topology
(`--size s`) is a handful of containers, but the larger ISP and CLOS fabrics are
tens of them. Start with `--size s` and watch Docker Desktop's memory before
scaling up.
