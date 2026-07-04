# Deploying Quên on Alibaba Cloud

The engine, API and dashboard ship as **one process** — `quen-api` serves the
built dashboard from the same origin (`QUEN_DASHBOARD_DIST`), so a single
small ECS instance is all a demo needs. No nginx, no CORS, no database
server (SQLite WAL on local disk).

## Fully scripted path (RAM AccessKey → zero console clicks)

With a RAM key that has `AliyunECSFullAccess` + `AliyunVPCFullAccess`
(`pip install alibabacloud_ecs20140526 alibabacloud_vpc20160428` first):

```bash
export ALIBABA_CLOUD_ACCESS_KEY_ID=...  ALIBABA_CLOUD_ACCESS_KEY_SECRET=...
python deploy/provision.py                 # security group + instance +
                                           # cloud-init bootstrap (OFFLINE demo)
python deploy/provision.py --enable-live   # then: DashScope key via Cloud
                                           # Assistant (never via user_data —
                                           # instance metadata is readable by
                                           # any process on the box)
```

Idempotent: re-running reports the existing `quen-demo` instance. Delete the
RAM key after deploying.

## The 5-minute path (ECS console → one command)

1. **Console** ([ecs.console.aliyun.com](https://ecs.console.aliyun.com)) →
   *Create Instance*:
   - Region: **Singapore (ap-southeast-1)** — same side of the world as the
     DashScope international endpoint used by `alibaba_client.py`.
   - Instance: any 2 vCPU / 2 GiB burstable (e.g. `ecs.e-c1m1.large` or
     `ecs.t6-c1m2.large`), pay-as-you-go.
   - Image: **Ubuntu 24.04** (22.04 fine too).
   - Public IP: assign; bandwidth pay-by-traffic is cheapest.
   - Security group: allow inbound **TCP 22** (SSH) and **TCP 80** (HTTP).
2. **Connect** (console "Workbench" button, or `ssh root@<ip>`), then:

   ```bash
   git clone -b claude/quen-memory-agent-spec-p3izss \
     https://github.com/phamthanhhang208/quen.git /opt/quen
   # optional — live Qwen mode; skip for the zero-cost offline demo:
   printf 'DASHSCOPE_API_KEY=sk-...\n' > /opt/quen/.env && chmod 600 /opt/quen/.env
   sudo bash /opt/quen/deploy/setup.sh
   ```

3. Open `http://<ip>/` — the dashboard, seeded with the full 80-day demo
   narrative (supersession, eviction, verification all visible).
   Health check: `http://<ip>/vitals`.

`setup.sh` is idempotent — after a `git pull`, re-run it (or just
`systemctl restart quen`). Logs: `journalctl -u quen -f`.

## Modes

| | offline (default) | live |
|---|---|---|
| trigger | no `.env` | `/opt/quen/.env` with `DASHSCOPE_API_KEY` |
| LLM/embeddings | scripted + hashing (deterministic, $0) | qwen-flash / qwen3.5-plus / text-embedding-v4 |
| good for | judges clicking through the demo narrative | ingesting/asking real content on stage |

The demo database is seeded either way; live mode only changes what powers
*new* `/ingest` and `/ask` calls. To re-seed live end-to-end (~$0.01):
`.venv/bin/python scripts/seed_demo.py --live data/demo.db && systemctl restart quen`.

## Alternatives (not scripted here)

- **Function Compute / SAE**: works (custom runtime, `quen-api` as the HTTP
  handler) but SQLite state is per-instance and cold starts hurt a demo;
  an ECS box is simpler and steadier for judging week.
- **OSS static hosting + separate API host**: build the dashboard with
  `VITE_API_BASE=https://api.example.com npm run build` and add that origin
  to the CORS list in `src/quen/api.py`.
