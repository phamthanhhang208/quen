# Quên — 3-minute demo video script (target length ≈ 2:30)

**Chuẩn bị trước khi quay (5 phút):**

```bash
QUEN_OFFLINE=1 QUEN_DB_PATH=data/demo.db QUEN_VERIFY_REPO=scripts/demo_repo .venv/bin/quen-api
cd dashboard && npm run dev        # http://localhost:5173
```

(hoặc mở URL máy ECS nếu đã deploy). Trình duyệt full-screen, zoom 110–125%,
tắt notification. Mở sẵn các tab theo thứ tự cảnh. Quay màn hình bằng
OBS/QuickTime, đọc voiceover theo từng cảnh — mỗi cảnh một lần chuyển tab,
không cần chuyển cảnh cầu kỳ.

---

## Cảnh 1 · 0:00–0:20 · Hook — mở README (phần title + câu "Why")

> Agent memory doesn't fail by forgetting too much. It fails by being
> **confidently wrong from stale memory**. This is **Quên** — Vietnamese for
> "to forget", sounds like *Qwen* — a Qwen-powered memory that knows **what
> to forget, and how sure to sound**.

## Cảnh 2 · 0:20–0:45 · Tab **Memories** → click một memory mở **Memory detail**

*(chỉ chuột vào cột R đang giảm, các chip trạng thái; trong detail: đường
forgetting curve với các mốc review và ngưỡng eviction θ)*

> Eighty days of a team's memory. Every fact carries real FSRS retention
> state — you can watch retrievability decay live. Reviews bend the curve;
> the red line is the eviction threshold.

## Cảnh 3 · 0:45–1:10 · Tab **Dream log** (mở run có supersession + run có eviction)

*(chỉ vào badge `deterministic` của cặp useApi → useQuery, rồi entry eviction)*

> Consolidation runs offline, like sleep. When a PR migrated the team from
> useApi to useQuery, the **deterministic slot rule** caught it — no LLM
> call, fully auditable. Augmentations never supersede. Unused trivia gets
> **evicted** — low retention, past TTL, not pinned. Nothing is ever
> hard-deleted: tombstones only.

## Cảnh 4 · 1:10–1:50 · Tab **Recall trace** (trace cuối cùng — đây là cảnh đinh)

*(lần lượt chỉ: chip trust, sự kiện verify CONFIRMED, sự kiện REFUTED của
uploadV1 ở trace "uploads", mục excluded "superseded by…", rồi bật toggle
counterfactual)*

> At answer time, **trust is a runtime decision**. This aged fact fell below
> the trust gate, so before answering, Quên **verified it against the live
> repo**: confirmed — confidence rises, and the outcome feeds back into
> retention as an FSRS review. That's the closed loop. This other memory was
> **refuted** — tombstoned on the spot. And this toggle shows the
> counterfactual: the stale fact a plain RAG **would have injected**.

## Cảnh 5 · 1:50–2:15 · README — bảng Results (probe + LongMemEval)

*(scroll chậm qua 2 bảng và chart budget curve)*

> We measure the honesty. On a code-staleness probe: **FAMA 0.93 versus 0.40**
> for append-only RAG — p equals three times ten to the minus five — robust
> to paraphrasing, at fewer delivered tokens. And on LongMemEval, where
> nothing ever goes stale, we report **where we lose**. Forgetting is a tax
> on stale-free recall — and a big win the moment the world changes.

## Cảnh 6 · 2:15–2:30 · README — mục MCP server

> Everything runs on Qwen via Alibaba Cloud DashScope — the full eval cost
> under two dollars. Quên ships as an MCP server: plug it into any agent.
> **Remember what matters. Forget what's stale. Verify before you assert.**

---

**Mẹo:** đọc chậm hơn bình thường một chút; tổng thoại ~300 từ ≈ 2:10–2:30.
Nếu cần cắt xuống 2:00: bỏ câu "Reviews bend the curve…" (cảnh 2) và câu
"Augmentations never supersede." (cảnh 3).
