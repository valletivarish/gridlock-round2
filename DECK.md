# Gridlock Hackathon 2.0 — Round 2 Pitch Deck
**PS2: Event-Driven Congestion | Bengaluru Traffic Police**
Flipkart Finale · Jul 3, Flipkart HQ

---

## Slide 1 — Title

**Event Intelligence for BTP**
*From Reactive Response to Proactive Deployment*

- Dataset: 8,173 real ASTraM events, Bengaluru, Nov 2023 – Apr 2024
- Three pillars: Forecast Impact → Recommend Resources → Keep Learning
- Live prototype: a Streamlit dashboard with a real-time Event Simulator
- Audience: Bengaluru Traffic Police + Flipkart leadership
- No external data. Every number on every slide traces back to BTP's own ASTraM records.

*Speaker note: Lead with the key promise — BTP gets actionable numbers the moment an event is logged, not hours later.*

---

## Slide 2 — The Problem: Three Gaps BTP Faces Today

**BTP's own brief names three operational gaps. We built exactly one pillar per gap.**

| BTP Gap | What this means in practice |
|---|---|
| "Event impact not quantified in advance" | Officers arrive with no estimate of how long the event will last or whether the road will close |
| "Resource deployment is experience-driven" | Manpower decisions depend on individual judgment — inconsistent across shifts and zones |
| "No post-event learning system" | Each event is forgotten once cleared; patterns repeat; the system never improves |

- 94% of all events are unplanned — reactive response is the default mode
- 5,030 events (62%) carry a High priority flag demanding immediate response
- Without quantified impact, even high-priority events compete for the same constrained officer pool

*Speaker note: Frame the three gaps as BTP's words, not ours — this is solving the problem statement verbatim.*

---

## Slide 3 — The Data: Real BTP Records, No Synthetic Fabrication

**8,173 ASTraM events · 6 months · entire city · used as-is from the provided dataset**

- Time span: November 2023 – April 2024 (6 full months)
- Coverage: all corridors, zones, junctions, and priority levels across Bengaluru
- 116 malformed timestamps caught and corrected in pre-processing — no silent data loss
- Fields used at inference time: event cause, type, corridor, zone, junction, police station, vehicle type, priority, latitude/longitude, hour, day-of-week, month
- Only provided ASTraM data used — no external feeds, no synthetic rows

*Speaker note: Reassure judges the numbers are auditable — every figure links to a specific cell in the dataset or EDA table.*

---

## Slide 4 — Key Insights: What the Data Actually Says

**Four findings that directly shape our solution design.**

`[fig_04_corridors.png — top corridor bar chart]`
`[fig_05_resolution_by_cause.png — resolution time by cause]`
`[fig_06_drift.png — event-cause drift over 6 months]`

- **60% of all events are vehicle breakdowns** (4,896/8,173) — concentrated on Mysore Road (743 events) and Bellary Road 1 (610 events), both 100% high-priority
- **Construction is a 5-hour bottleneck** — median clearance 296 min vs 41 min for breakdowns; P75 stretches to 427 min
- **80% of events occur outside peak hours** — the single busiest hour is 2 AM (845 events, mostly uncleared breakdowns that become the 8 AM gridlock)
- **Event mix is drifting**: vehicle-breakdown share fell from 66% to 49% over 6 months — a static model degrades quietly without a learning loop

*Speaker note: The 2 AM finding is the counterintuitive hook — night-shift coverage is not optional, it is the highest-volume hour in six months of data.*

---

## Slide 5 — Our Solution: Three Pillars + One Dashboard

**One input (the event report). Three outputs (impact, resources, learning). One interface.**

```
Event logged in ASTraM
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  PILLAR 1: Forecast Impact                                  │
│  Road-closure probability (AUC 0.816 on future data)        │
│  Expected clearance time (EDA-benchmarked per cause)        │
│  Severity: Low / Medium / High                              │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PILLAR 2: Recommendation Engine                            │
│  Officer count │ Barricade flag │ Diversion route           │
│  Nearest responding police station                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PILLAR 3: Post-Event Learning Loop                         │
│  Monthly retrain on new resolved events                     │
│  Drift monitoring — alert if AUC drops below 0.70           │
└─────────────────────────────────────────────────────────────┘
```

- Streamlit dashboard: live map, hotspot heatmap, trend charts, Event Simulator
- Event Simulator lets judges enter any event live and see recommendations instantly
- All three pillars run offline — no external API dependency

*Speaker note: Demo the Event Simulator here — type in a construction event on Mysore Road at 9 AM and show the output on screen.*

---

## Slide 6 — Pillar 1: Impact Forecasting (Honest Model Results)

**CatBoost, time-based train/test split: trained on past, tested on future months.**

`[fig_07_learning.png — AUC over time, static vs retrained]`

**Road-closure classifier (CatBoost, the primary signal)**

| Validation split | ROC-AUC | F1 |
|---|---|---|
| Future months (Mar–Apr 2024) | **0.816** | 0.450 |
| Unseen corridors (6 held out) | **~0.70 (0.696)** | 0.335 |
| Cold-start (unseen corridor + junction) | **~0.73 (0.731)** | — |

- AUC 0.816 on months the model has never seen — this is the honest "does it work tomorrow?" number
- AUC ~0.70 on entirely unseen corridors — degrades from the time-split result but stays above random; still operationally useful
- F1 is lower because road closures are rare (8.3% of events) — AUC is the right metric for a skewed binary target
- Top feature by importance: event cause (36 points) — the system can identify closure risk from cause alone even for an unknown corridor

**Duration benchmark (CatBoost, MAE on future data)**

- MAE = 101 min on future test set — honest planning uncertainty, not a flaw
- Duration is right-skewed (median 46 min, max 1,437 min); the benchmark serves as a planning guide, not a stopwatch
- For field communication, EDA-derived cause-level medians (vehicle breakdown: 41 min; construction: 296 min) are cleaner and safer than point predictions

*Speaker note: State the F1 limitation proactively — judges will ask; owning it builds more credibility than hoping they don't notice.*

---

## Slide 7 — Pillar 2: Recommendation Engine — A Worked Example

**What a field officer sees within seconds of logging an event.**

**Scenario: Construction on Mysore Road, 9 AM, High priority**

```
Expected clearance      ~296 min  (EDA: construction median — nearly 5 hours)
Road-closure prob        62%       (model output)
Severity                 HIGH      (all four signals fire: high prob + High priority
                                   + top-10 hotspot + slow-clearing cause)
Recommended officers     10        (base 6 + 2 peak-hour + 2 closure bonus)
Barricading              YES       (62% > 35% threshold = 4x base-rate uplift)
Diversion                Magadi Road or Chord Road
Responding station       Halasuru Gate
Rationale                High-impact event on city's #1 congestion corridor;
                         road closure likely; expect ~5 hours; peak hour.
```

- Officer count logic: severity tier sets the base (Low=2, Medium=4, High=6); peak-hour and closure probability add structured bonuses — no black box
- Barricade trigger set at 35% model probability = 4x the historical base rate of 8.3%
- Station lookup uses a three-layer fallback: corridor history → zone history → Haversine nearest — never crashes on unseen locations
- All outputs are plain English — designed for a field officer, not a data scientist

*Speaker note: Walk through the rationale field specifically — this is the explainability story that matters to police leadership.*

---

## Slide 8 — Pillar 3: The Learning Loop

**A static model is a liability. The event mix is already shifting.**

`[fig_06_drift.png — breakdown share drifting 66% → 49%]`
`[fig_07_learning.png — static vs retrained AUC month by month]`

| Month tested | AUC (Static, frozen model) | AUC (Retrained monthly) | Difference |
|---|---|---|---|
| Jan 2024 | 0.751 | 0.747 | −0.004 |
| Feb 2024 | 0.696 | 0.714 | **+0.018** |
| Mar 2024 | 0.792 | 0.805 | **+0.013** |
| Apr 2024 | 0.822 | 0.838 | **+0.016** |

- February is the clearest failure case: the static model drops to AUC 0.696 while retraining holds it at 0.714 — a meaningful difference on a rare-class prediction problem
- Retraining never makes things materially worse (Jan: −0.004, negligible)
- Recommended cadence: **monthly** — aligns with the observed cause-mix shift; weekly is overkill at current data volumes; quarterly risks another Feb-style dip
- Implementation: new resolved events appended → CatBoost retrained → AUC logged → alert if AUC < 0.70

*Speaker note: The Feb dip is the concrete argument for the learning loop — one month of frozen weights and the model noticeably weakens.*

---

## Slide 9 — Robustness: Works on Data the Model Has Never Seen

**Three validation dimensions, not one — because production is always the future.**

| Dimension | Test | Result |
|---|---|---|
| Future time | Train Nov–Feb; test Mar–Apr | AUC 0.816 (closure), MAE ~103 min, R² ≈ 0 (duration) |
| Unseen locations | Hold out 6 corridors entirely | AUC ~0.70 / 0.696 (closure); cold-start ~0.73 / 0.731 |
| Unseen causes/corridors | Fog on NH-44 Bypass (never in training) | Degrades gracefully: zone fallback, AUC-estimated closure, 41-min benchmark |

**Robustness built into every layer:**
- Out-of-fold target encoding (alpha=20 smoothing) prevents data leakage and handles new corridors at inference without crashing
- CatBoost native categorical handling absorbs completely new cause labels at inference time
- Three-layer station lookup (corridor → zone → Haversine) ensures a valid station is always returned
- 116 malformed timestamps caught and fixed during ingestion — the pipeline is tolerant of real-world messy data

`[fig_07_learning.png — consistent AUC across held-out months]`

*Speaker note: Robustness is the most technically credible pillar — stress that the model was never shown March or April when trained.*

---

## Slide 10 — Real-World Impact & Scalability for BTP

**What changes the day BTP deploys this system.**

**Operational changes — immediate:**
- Every event logged gets a severity label and officer count recommendation within seconds
- High-priority closure-risk events (62%+ closure probability on Mysore Road construction) trigger automatic barricade staging and diversion activation
- Night-shift coverage gap is quantified: 845 events at 2 AM need staffed response, not morning reaction
- Seasonal planning: February–March surge (peak: 1,956 events in March) can be resourced in advance

**Scalability path — honest assessment:**
- Current model: 8,173 events over 6 months; retraining takes under 5 minutes on a laptop
- Handles new corridors, causes, and junctions without code changes — via built-in fallbacks
- Adding real-time ASTraM feed: replace the CSV loader with an API call; all logic downstream is unchanged
- Multi-city extension: model re-trains from scratch on a new city's ASTraM data; no hardcoded Bengaluru assumptions except the station fallback (Cubbon Park) — one config change

**What this is not:**
- Not a replacement for field judgment — officer counts are starting points; supervisors adjust for ground conditions
- Not a real-time traffic signal controller — it advises human dispatchers, it does not automate deployment

*Speaker note: Emphasise the "advises, does not automate" framing — BTP retains authority over every deployment decision.*

---

## Slide 11 — The Ask

**One prototype ready now. Three things needed to make it operational.**

**What we've built:**
- Working prototype: notebook (EDA + models + learning loop) + Streamlit dashboard with live Event Simulator
- All models trained and saved; inference runs in under 1 second per event
- Monthly retraining script ready to schedule

**The ask from BTP:**
1. **Access to a live ASTraM feed** — to move from historical replay to real-time recommendations; the pipeline is already built for it
2. **Feedback labels from field officers** — "was the closure prediction right? was manpower sufficient?" — to close the learning loop with ground truth
3. **Pilot corridor** — run the recommendation engine alongside manual dispatch on Mysore Road for 30 days; compare officer utilisation and response time

**The ask from Flipkart:**
- Infrastructure and deployment support for the Streamlit dashboard on BTP's internal network

*Speaker note: Close with a concrete, modest ask — not "deploy citywide" but "one corridor, one month." That is credible and actionable.*

---

*All figures referenced: fig_01_monthly_trend.png · fig_02_hourly_profile.png · fig_03_dow.png · fig_04_corridors.png · fig_05_resolution_by_cause.png · fig_06_drift.png · fig_07_learning.png*

*All numbers traceable to: round_2/eda/INSIGHTS.md · round_2/models/MODELS.md · round_2/RECOMMEND.md · round_2/LEARNING.md*
