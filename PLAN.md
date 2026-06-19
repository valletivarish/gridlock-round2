# Gridlock Hackathon 2.0 — Round 2 Plan

**Theme 2: Event-Driven Congestion (Planned & Unplanned)** · Solo · Deadline **Jun 21, 11:59 PM IST** · Finale Jul 3 (Flipkart HQ)

**Dataset:** ASTraM event data (provided) — 8,173 real Bengaluru traffic events. **Use ONLY this dataset (external data → disqualification).**

---

## What PS2 officially requires (the 3 gaps → our 3 pillars)

| BTP's stated gap | What we must deliver |
|------------------|----------------------|
| "Event impact is **not quantified in advance**" | **Pillar 1 — Forecast event impact** (duration, severity, road-closure) |
| "Resource deployment is **experience-driven**" | **Pillar 2 — Recommend manpower, barricading, diversion** |
| "**No post-event learning** system" | **Pillar 3 — Post-event learning loop** |

Judged on: innovation · prototype clarity · robustness · scalability · real-world viability.

---

## Robustness to unseen / production / drifting data (cross-cutting must-have)

The prototype must hold up on data it has never seen — new events, new locations, format
changes, and drift over time. Baked in everywhere:
- **Robust ingestion** — tolerant timestamp parsing (varying precision), missing-value
  handling, no crashes on messy input. *(✓ already tested — caught & fixed 116 bad timestamps)*
- **Unseen categories** — new corridors / causes / junctions fall back to global priors
  (out-of-fold target encoding) + native categorical handling → never break.
- **Honest validation = TIME split** — train on earlier months, test on later months →
  measures real performance on *future / unseen* events (the drift case). Plus
  group-by-corridor test for *unseen locations*.
- **Drift adaptation** — the post-event learning loop (Pillar 3) updates the model as new
  events resolve.
- **Edge-case tests** — quantify generalization to unseen time periods / corridors / causes,
  reported honestly (same discipline as Round 1).

---

## TODO checklist

### Phase 0 — Setup & compliance
- [ ] Create `round_2/` structure (`data/`, notebook, `app/`, `deck/`)
- [ ] Install env: streamlit, folium/plotly (lightgbm, catboost, pandas already present)
- [ ] Lock rule: **only the provided ASTraM dataset** — no external data

### Phase 1 — Data understanding & cleaning
- [ ] Load ASTraM events; parse all timestamps (start / resolved / closed / created)
- [ ] Compute event **duration**; cap extreme outliers (p95 = noise)
- [ ] Standardize `event_type`, `event_cause`, `corridor`, `zone`, `junction`, `priority`, `requires_road_closure`
- [ ] Validate lat/lon are within Bengaluru; handle missing fields
- [ ] Short data-quality note

### Phase 2 — Insight / EDA (becomes the pitch story)
- [ ] Patterns by type (planned vs unplanned), cause, priority
- [ ] **Spatial hotspots:** worst corridors / junctions / zones (frequency × impact)
- [ ] **Temporal:** time-of-day, day-of-week, monthly trends
- [ ] **Resolution bottlenecks:** which causes / corridors clear slowest
- [ ] Road-closure drivers
- [ ] Deck-ready charts + a hotspot map

### Phase 3 — PILLAR 1: Forecast event impact *(quantify in advance)*
- [ ] **Duration model** (regression) — predicted minutes to clear
- [ ] **Severity / road-closure model** (classification) — likelihood + priority
- [ ] Features: cause, type, location/corridor/junction, time-of-day, vehicle type
- [ ] Honest **time-aware** validation; report MAE/R² and AUC/F1
- [ ] Feature importance (explainability for the panel)

### Phase 4 — PILLAR 2: Resource recommendation engine *(manpower / barricading / diversion)*
- [ ] **Manpower** recommendation (predicted impact + priority + corridor)
- [ ] **Barricading** recommendation (road-closure prediction + cause)
- [ ] **Diversion** suggestion (corridor + alternate route / nearby junction)
- [ ] **Nearest police station / zone** routing from location
- [ ] Logic grounded in observed data patterns, not arbitrary rules

### Phase 5 — PILLAR 3: Post-event learning loop *(no-learning gap)*
- [ ] Log predicted vs actual outcome per event
- [ ] Update/retrain models as new events resolve
- [ ] Demonstrate accuracy improving as it "learns"

### Phase 6 — Dashboard + live event simulator (the demo)
- [ ] Map view: events + risk-hotspot heatmap
- [ ] **Event simulator:** enter a new event → predicted impact + recommended response (live, in front of judges)
- [ ] Analytics views: trends, corridor rankings, bottlenecks
- [ ] Clean, demo-ready Streamlit UI

### Phase 7 — Pitch package
- [ ] **[REQUIRED]** Working prototype (notebook + app code)
- [ ] **[REQUIRED]** Deck / approach write-up: problem → data → insights → models → recommendation → **BTP impact** → scalability
- [x] **Demo video — SKIPPED** (not required by the brief)
- [ ] README + `requirements.txt`

### Phase 8 — Submit (compliance)
- [ ] **Check the "Submit Now" form for the exact mandatory fields** (the brief doesn't list them)
- [ ] Every file **< 50 MB**
- [ ] Bundle: notebook + app code + deck (+ video if doing it)
- [ ] Select **Theme 2 (Event-Driven Congestion)** on HackerEarth
- [ ] Submit before **Jun 21, 11:59 PM IST**

---

## "Real-time data" note
The system is built on historical data but designed to take a **live event as input** (the simulator) — satisfying the "historical **and real-time** data" wording in the problem statement.
