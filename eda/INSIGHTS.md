# BTP ASTraM — Event Intelligence: Headline Findings

*Dataset: 8,173 real BTP operational events, Bengaluru, Nov 2023 – Apr 2024.*

---

## 1. Unplanned incidents dominate — and they're mostly high-priority

**7,706 of 8,173 events (94%) are unplanned.**
5,030 events (62%) carry a High priority flag, meaning more than half the city's traffic incidents demand an immediate BTP response.
Only 467 events (5.7%) are planned activities where BTP can pre-position resources.

**Takeaway:** The city's congestion story is reactive, not proactive. Predictive deployment beats after-the-fact response.

---

## 2. Vehicle breakdowns drive 60 % of all events — and cluster on the same roads

**4,896 events (60% of total) are vehicle breakdowns** — far ahead of potholes (537),
construction (480), and water-logging (458).
These breakdowns concentrate on a handful of corridors (see #3).

**Takeaway:** Targeting rapid tow/clearance on the top 3–5 corridors would resolve the majority of congestion triggers.

---

## 3. Three corridors absorb most of the impact

Ranked by a composite score (frequency × median resolution time × high-priority share × road-closure rate):

| Rank | Corridor | Events | Median resolution | HP rate | Road closures |
|------|----------|--------|-------------------|---------|---------------|
| 1 | Mysore Road | 743 | 41 min | 100% | 11% |
| 2 | Bellary Road 1 | 610 | 42 min | 100% | 5% |
| 3 | Airport New South Road | 67 | 60 min | 100% | 10% |

**Takeaway:** A dedicated, always-on rapid-response post on Mysore Road and Bellary Road 1 would address the highest-burden corridors.

---

## 4. Road closures are rare but severe — 676 events force full blockages

**676 events (8.3%) require a road closure**, affecting all traffic on that stretch.
These closures are concentrated in construction and planned-event categories.

**Takeaway:** Pre-authorising alternative routing notifications for closure-tagged events can cut downstream cascade congestion.

---

## 5. Construction and road conditions are the slowest to clear

| Cause | Median clearance | P75 clearance |
|-------|-----------------|---------------|
| Construction | 296 min | 427 min |
| Road conditions | 246 min | 756 min |
| Vehicle breakdown | 41 min | 73 min |

Construction incidents take a **median 296 minutes** — nearly 5× longer than a typical breakdown.
Road-condition reports (246 min median) are the second worst bottleneck.

**Takeaway:** Separate escalation workflows for infrastructure-class events (construction, road conditions) rather than treating them as standard incidents.

---

## 6. The single busiest hour is 2 AM — off-peak events far outnumber peak-hour events

Peak-hour events (8–10 AM and 5–8 PM): **1,616 events (20%).**
**80% of events fall outside defined peak hours.** The busiest single hour is **2:00 AM with 845 events** —
confirmed real night-shift incidents (488 vehicle breakdowns, dominated by ORR East 2 and Mysore Road).
These breakdowns sit uncleared through the night and feed morning gridlock.

**Takeaway:** Night-shift BTP coverage is not a nice-to-have — it is the largest single-hour load in the dataset.
Uncleared 2 AM breakdowns become the 8 AM congestion problem.

---

## 7. March 2024 was the highest-stress month — 1956 events in one month

Monthly volumes: Nov-23 (953) → Dec-23 → Jan-24 → Feb-24 → **Mar-24 (1956)** → Apr-24.
March 2024 showed a surge, possibly seasonal (summer heat + road works before monsoon).

**Takeaway:** Seasonal staffing plans should pre-load resources in February–March, not react to the spike after it starts.

---

## 8. Event mix is shifting — vehicle-breakdown share changed from 66% to 49%

Between 2023-11 and 2024-04, the monthly breakdown share moved from **66%** to **49%**.
This kind of drift in event mix means static thresholds and fixed resource rules will gradually go stale.

**Takeaway:** The Event Intelligence system must be retrained/recalibrated at least monthly to track real-world drift — a static model is a liability.

---

*Files: fig_01_monthly_trend.png · fig_02_hourly_profile.png · fig_03_dow.png · fig_04_corridors.png · fig_05_resolution_by_cause.png · fig_06_drift.png · table_top10_corridors.csv · table_top15_junctions.csv · table_bottleneck_causes.csv*
