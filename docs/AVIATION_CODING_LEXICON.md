# AVIATION CODING LEXICON

Complete reference for NTSB aviation accident coding system. Extracted from `ref_docs/codman.pdf` (Revised December 1998).

## Table of Contents

- [Overview](#overview)
- [Coding System Structure](#coding-system-structure)
- [Section I: Occurrences (100-430)](#section-i-occurrences-100-430)
- [Section II: Phase of Operation (500-610)](#section-ii-phase-of-operation-500-610)
- [Section IA: Aircraft/Equipment Subjects (10000-21104)](#section-ia-aircraftequipment-subjects-10000-21104)
- [Section IB: Performance/Operations (22000-25000)](#section-ib-performanceoperations-22000-25000)
- [Section II: Direct Underlying Causes (30000-84200)](#section-ii-direct-underlying-causes-30000-84200)
- [Section III: Indirect Underlying Causes (90000-93300)](#section-iii-indirect-underlying-causes-90000-93300)
- [Code Usage in Analysis](#code-usage-in-analysis)

## Overview

The NTSB uses a hierarchical coding system to classify aviation accidents and incidents. This system enables:

- **Systematic categorization** of accident events
- **Statistical analysis** of causal factors
- **Pattern recognition** across accident types
- **Regulatory trend analysis**
- **Machine learning feature engineering**

**Coding Manual**: `ref_docs/codman.pdf` (comprehensive 124-page reference)

## Coding System Structure

```
Occurrences (100-430)
  ├── What happened
  └── Observable events

Phase of Operation (500-610)
  ├── When it happened
  └── Flight phase

Aircraft/Equipment (10000-21104)
  ├── Airframe (10000-11700)
  ├── Systems (12000-13500)
  └── Powerplant (14000-17710)

Performance/Operations (22000-25000)
  ├── Performance subjects (22000-23318)
  └── Operations subjects (24000-25000)

Direct Causes (30000-84200)
  ├── Why it happened (root causes)
  └── Organized by equipment type

Indirect Causes (90000-93300)
  ├── Contributing factors
  └── Design, maintenance, organizational
```

## Section I: Occurrences (100-430)

Observable events that occurred during the accident sequence.

### Engine/Propulsion (100-180)

| Code Range | Category | Examples |
|------------|----------|----------|
| 100-110 | Engine Failure | Total failure, partial failure, misfiring |
| 120-130 | Powerplant System Failure | Fuel system, ignition, cooling |
| 140-150 | Propeller/Rotor Malfunction | Separation, overspeed, pitch control |
| 160-170 | Engine Fire/Explosion | In-flight fire, ground fire, explosion |

**Key Codes**:
- **100**: Engine Failure - Complete loss of power
- **110**: Engine Failure - Partial loss of power
- **140**: Propeller/rotor separation
- **150**: Propeller/rotor blade damage
- **160**: Engine fire/explosion

### Flight Control (200-250)

| Code Range | Category | Examples |
|------------|----------|----------|
| 200-210 | Flight Control Failure | Aileron, elevator, rudder malfunction |
| 220-230 | Control Surface Separation | In-flight breakup component |
| 240-250 | Flight Control Jamming | Binding, restricted movement |

### Structural (260-290)

| Code | Description | Significance |
|------|-------------|--------------|
| 260 | In-flight structural failure | Catastrophic breakup |
| 270 | Wing/stabilizer separation | Critical structural loss |
| 280 | Landing gear collapse | Ground handling accident |
| 290 | Airframe icing | Weather-related structural load |

### Collision (300-370)

| Code | Description | Context |
|------|-------------|---------|
| 300 | Midair collision | Two aircraft in flight |
| 310 | Collision with terrain | CFIT (Controlled Flight Into Terrain) |
| 320 | Collision with obstacle | Tower, wire, tree |
| 330 | Collision with water | Water impact (ditching/crash) |
| 340 | Ground collision | Taxiway/runway collision |
| 350 | Object struck on ground | FOD, vehicle, equipment |

**Critical**: Code 310 (CFIT) is major focus for prediction models.

### Fuel Related (380-390)

| Code | Description | Preventability |
|------|-------------|----------------|
| 380 | Fuel exhaustion | Completely out of fuel - highly preventable |
| 390 | Fuel starvation | Fuel available but not reaching engine |

### Fire/Smoke (400-410)

| Code | Description |
|------|-------------|
| 400 | Fire - in-flight |
| 405 | Fire - post-crash |
| 410 | Smoke - in-flight |

### Other Occurrences (420-430)

| Code | Description |
|------|-------------|
| 420 | Hard landing |
| 425 | Abrupt maneuver |
| 430 | Loss of control - in flight |

**Code 430** (Loss of Control) is most common fatal accident occurrence.

## Section II: Phase of Operation (500-610)

Identifies when in the flight the occurrence happened.

### Ground Operations (500-540)

| Code | Phase | Description |
|------|-------|-------------|
| 500 | Standing | Aircraft stationary, engines off/on |
| 510 | Taxi | Ground movement to/from runway |
| 520 | Taxi to runway | Pre-flight taxi |
| 530 | Taxi from runway | Post-flight taxi |

### Takeoff Phase (550-570)

| Code | Phase | Critical Period |
|------|-------|----------------|
| 550 | Takeoff - initial climb | Most critical phase |
| 560 | Takeoff - aborted | Rejected takeoff |

### En Route (580-585)

| Code | Phase | Duration |
|------|-------|----------|
| 580 | Climb to cruise | Transitioning to cruise altitude |
| 582 | Cruise | Level flight at cruise altitude |
| 585 | Descent | Descending from cruise |

### Approach & Landing (590-600)

| Code | Phase | Risk Level |
|------|-------|------------|
| 590 | Approach | Second most critical phase |
| 595 | Go-around | Missed approach |
| 600 | Landing | High accident rate phase |
| 605 | Landing - flare/touchdown | Critical sub-phase |
| 610 | Landing - roll | Post-touchdown |

### Maneuvering (620)

| Code | Phase | Context |
|------|-------|---------|
| 620 | Maneuvering | Low-level maneuvers, aerobatics |

**Statistical Note**: Phases 550 (takeoff) and 590-600 (approach/landing) account for 70%+ of fatal accidents despite being <5% of flight time.

## Section IA: Aircraft/Equipment Subjects (10000-21104)

Hierarchical codes for aircraft components and systems.

### Airframe (10000-11700)

#### Wing Structure (10000-10500)

| Code Range | Component | Sub-components |
|------------|-----------|----------------|
| 10000-10100 | Wing - main structure | Spar, rib, skin |
| 10200-10300 | Wing - control surfaces | Aileron, flap, slat |
| 10400-10500 | Wing - attachments | Strut, brace |

#### Fuselage (10600-11000)

| Code Range | Component |
|------------|-----------|
| 10600-10700 | Fuselage - structure |
| 10800-10900 | Fuselage - doors/windows |
| 11000 | Fuselage - cabin/cargo |

#### Landing Gear (11100-11400)

| Code | Component | Failure Mode |
|------|-----------|--------------|
| 11100 | Main landing gear | Collapse, retraction failure |
| 11200 | Nose landing gear | Shimmy, collapse |
| 11300 | Tail gear | Rare, taildragger aircraft |
| 11400 | Landing gear control | Hydraulic, electrical |

#### Flight Controls (11500-11700)

| Code Range | System | Components |
|------------|--------|------------|
| 11500-11550 | Pitch control | Elevator, stabilator |
| 11600-11650 | Roll control | Ailerons, spoilers |
| 11700 | Yaw control | Rudder, trim tabs |

### Systems (12000-13500)

#### Hydraulic System (12000-12200)

| Code | Component |
|------|-----------|
| 12000 | Hydraulic pump |
| 12100 | Hydraulic reservoir |
| 12200 | Hydraulic lines/fittings |

#### Electrical System (12300-12500)

| Code | Component |
|------|-----------|
| 12300 | Alternator/generator |
| 12400 | Battery |
| 12500 | Electrical wiring |

#### Environmental System (12600-12800)

| Code | System |
|------|--------|
| 12600 | Heating system |
| 12700 | Air conditioning |
| 12800 | Pressurization |

#### Fuel System (13000-13500)

| Code | Component | Criticality |
|------|-----------|-------------|
| 13000 | Fuel pump | High |
| 13100 | Fuel tank | High |
| 13200 | Fuel line | High - leaks cause fires |
| 13300 | Fuel valve/selector | High - pilot error common |
| 13400 | Fuel quantity indicating | Medium - situational awareness |

### Powerplant (14000-17710)

#### Reciprocating Engine (14000-15500)

| Code Range | Component | Common Failures |
|------------|-----------|-----------------|
| 14000-14200 | Crankshaft assembly | Fatigue, fracture |
| 14300-14500 | Piston/cylinder assembly | Scoring, compression loss |
| 14600-14800 | Valve assembly | Sticking, burning |
| 15000-15200 | Ignition system | Magneto failure |
| 15300-15500 | Carburetor | Icing, mixture control |

#### Turbine Engine (16000-17000)

| Code Range | Component |
|------------|-----------|
| 16000-16200 | Compressor |
| 16300-16500 | Combustion chamber |
| 16600-16800 | Turbine |
| 16900-17000 | Accessory gearbox |

#### Propeller (17200-17500)

| Code | Component |
|------|-----------|
| 17200 | Propeller hub |
| 17300 | Propeller blade |
| 17400 | Propeller governor |
| 17500 | Propeller spinner |

## Section IB: Performance/Operations (22000-25000)

### Performance Subjects (22000-23318)

#### Aerodynamics (22000-22500)

| Code | Subject | Description |
|------|---------|-------------|
| 22000 | Stall | Exceeding critical angle of attack |
| 22100 | Spin | Aggravated stall with rotation |
| 22200 | Maneuvering speed | Exceeding Va |
| 22300 | Never exceed speed | Exceeding Vne |

**Code 22000 (Stall)** is primary cause in 25%+ of GA accidents.

#### Environmental (22600-23000)

| Code | Subject |
|------|---------|
| 22600 | Altitude - inadequate |
| 22700 | Airspeed - inadequate |
| 22800 | Weather - adverse |
| 22900 | Visibility - inadequate |
| 23000 | Density altitude |

#### Weight & Balance (23100-23200)

| Code | Subject | Impact |
|------|---------|--------|
| 23100 | Overweight | Performance degradation |
| 23200 | CG out of limits | Controllability loss |

### Operations Subjects (24000-25000)

#### Pilot Technique (24000-24300)

| Code | Subject | Prevention |
|------|---------|------------|
| 24000 | Pilot technique - improper | Training focus |
| 24100 | Flight planning - inadequate | Preflight emphasis |
| 24200 | Decision making - poor | ADM training |
| 24300 | Situational awareness - lack of | CRM training |

#### Procedures (24400-24700)

| Code | Subject |
|------|---------|
| 24400 | Checklist - not used |
| 24500 | Emergency procedure - improper |
| 24600 | Preflight inspection - inadequate |
| 24700 | Fuel management - improper |

#### ATC & Maintenance (25000)

| Code | Subject |
|------|---------|
| 25000 | Air traffic control |
| 25100 | Maintenance - inadequate |
| 25200 | Inspection - inadequate |

## Section II: Direct Underlying Causes (30000-84200)

Detailed root cause codes organized by aircraft component and failure mode. This section mirrors Section IA structure but adds **why** the component failed.

### Structure

```
Component (from Section IA)
  ├── Mechanical Failure (3xxxx)
  ├── Design Deficiency (4xxxx)
  ├── Maintenance Error (5xxxx)
  ├── Material Defect (6xxxx)
  └── Environmental Damage (7xxxx)
```

### Example: Engine Failure Causes (30000-34999)

| Code Range | Cause Category |
|------------|----------------|
| 30000-30999 | Mechanical failure - improper maintenance |
| 31000-31999 | Mechanical failure - wear/fatigue |
| 32000-32999 | Design deficiency |
| 33000-33999 | Material defect/failure |
| 34000-34999 | Environmental/operational damage |

**Note**: Full enumeration of 30000-84200 range available in `ref_docs/codman.pdf` (80+ pages).

## Section III: Indirect Underlying Causes (90000-93300)

Contributing factors that enabled or exacerbated the accident.

### Organizational Factors (90000-91000)

| Code | Factor |
|------|--------|
| 90000 | Company management - inadequate oversight |
| 90100 | Training program - deficient |
| 90200 | Maintenance program - inadequate |
| 90300 | Safety culture - deficient |

### Regulatory Factors (91000-92000)

| Code | Factor |
|------|--------|
| 91000 | FAA oversight - inadequate |
| 91100 | Regulations - inadequate |
| 91200 | Certification process - deficient |

### Human Factors (92000-93000)

| Code | Factor |
|------|--------|
| 92000 | Fatigue |
| 92100 | Stress/psychological |
| 92200 | Physical impairment |
| 92300 | Knowledge deficiency |
| 92400 | Communication failure |

### Environmental/External (93000-93300)

| Code | Factor |
|------|--------|
| 93000 | Weather information - inadequate |
| 93100 | Airport facilities - inadequate |
| 93200 | Terrain/obstacles |

## Code Usage in Analysis

### Machine Learning Features

**Binary Features** (presence/absence):
```python
has_engine_failure = (occurrence_code >= 100) & (occurrence_code <= 110)
has_fuel_issue = (occurrence_code.isin([380, 390]))
is_collision = (occurrence_code >= 300) & (occurrence_code <= 370)
```

**Categorical Features**:
```python
phase_category = {
    500-540: 'ground',
    550-570: 'takeoff',
    580-585: 'cruise',
    590-610: 'approach_landing',
    620: 'maneuvering'
}
```

### Statistical Analysis

**High-Risk Occurrence Patterns**:
- Codes 100-110 (engine failure) + 380/390 (fuel) = 40% of GA accidents
- Code 310 (CFIT) + 590-600 (approach/landing phase) = fatal accident cluster
- Code 430 (loss of control) + 22000 (stall) = primary training focus

### NLP Code Mapping

Map narrative text to codes using:
- **Keyword extraction**: "engine failure" → 100
- **Context analysis**: "ran out of fuel on approach" → [390, 590]
- **BERT classification**: Fine-tune SafeAeroBERT on code→narrative pairs

### Temporal Analysis

**Code frequency trends**:
```sql
SELECT ev_year, occurrence_code, COUNT(*) as frequency
FROM Occurrences
GROUP BY ev_year, occurrence_code
ORDER BY ev_year, frequency DESC;
```

### Geospatial Clustering

**Code-location correlation**:
```python
# Cluster accidents by code and geography
from sklearn.cluster import DBSCAN
X = df[['dec_latitude', 'dec_longitude', 'occurrence_code_encoded']]
clusters = DBSCAN(eps=0.5, min_samples=10).fit(X)
```

## References & Tools

**Primary Reference**: `ref_docs/codman.pdf` - NTSB Aviation Coding Manual (Rev. 12/98)

**Database Tables**:
- `Occurrences.occurrence_code` - Primary occurrence field
- `Occurrences.phase_code` - Phase of operation field
- `Findings.finding_code` - Causal factor codes (Section IA/IB)

**Analysis Scripts**:
- `examples/quick_analysis.py` - Code frequency analysis
- `examples/advanced_analysis.py` - Multi-code pattern detection

**Validation**:
```python
# Validate code ranges
assert 100 <= occurrence_code <= 430, "Invalid occurrence code"
assert 500 <= phase_code <= 620, "Invalid phase code"
assert 10000 <= finding_code <= 93300, "Invalid finding code"
```

---

**Last Updated**: January 2025
**Version**: Based on NTSB Coding Manual Rev. 12/1998
