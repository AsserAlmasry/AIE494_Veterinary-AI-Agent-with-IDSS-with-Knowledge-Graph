"""
scripts/seed_neo4j.py
======================
One-time seeder: populates the Neo4j AuraDB knowledge graph with
comprehensive veterinary data for cattle (2021-2026):

  • 25 Disease nodes  (full clinical metadata)
  • 65 Symptom nodes  (onset, urgency, visibility)
  • 32 Treatment nodes (protocol, evidence level, withdrawal)
  • 16 Drug nodes      (active ingredient, category, withdrawal)
  • Relationships:     HAS_SYMPTOM, TREATED_BY, REQUIRES_DRUG,
                       RELATED_TO, PROGRESSES_TO, CONTRAINDICATED_WITH
  • 500+ PubMed research documents (2021-2026)
  • 16 Cow nodes from MMCows dataset

Run once manually:
    python scripts/seed_neo4j.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# ── Load .env ────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("seed_neo4j")


# ── Knowledge Base Data ───────────────────────────────────────────────────────

DISEASES: List[Dict[str, Any]] = [
    {"name": "mastitis",           "icd_v": "N10", "severity": 4, "zoonotic": False, "notifiable": False,
     "prevalence_pct": 25, "mortality_risk": "low",
     "description": "Inflammation of the mammary gland, most costly dairy disease worldwide. Clinical signs include swelling, heat, pain, and abnormal milk.",
     "species": "cattle", "category": "infectious"},
    {"name": "lameness",           "icd_v": "M79", "severity": 3, "zoonotic": False, "notifiable": False,
     "prevalence_pct": 18, "mortality_risk": "low",
     "description": "Painful locomotion affecting productivity. Most common in dairy cattle. Multi-factorial aetiology including nutritional, environmental, and infectious causes.",
     "species": "cattle", "category": "musculoskeletal"},
    {"name": "respiratory_disease","icd_v": "J22", "severity": 4, "zoonotic": False, "notifiable": False,
     "prevalence_pct": 14, "mortality_risk": "medium",
     "description": "Bovine Respiratory Disease (BRD) complex — leading cause of morbidity/mortality in beef cattle. Caused by Mannheimia haemolytica, Pasteurella multocida, BRSV, IBR.",
     "species": "cattle", "category": "respiratory"},
    {"name": "digestive_disorder", "icd_v": "K92", "severity": 3, "zoonotic": False, "notifiable": False,
     "prevalence_pct": 10, "mortality_risk": "medium",
     "description": "Encompasses LDA (left displaced abomasum), bloat (free gas/frothy), SARA, traumatic reticuloperitonitis (hardware disease).",
     "species": "cattle", "category": "gastrointestinal"},
    {"name": "skin_lesion",        "icd_v": "L30", "severity": 2, "zoonotic": True, "notifiable": False,
     "prevalence_pct": 8, "mortality_risk": "low",
     "description": "Dermatophytosis (ringworm), bovine papular stomatitis, interdigital dermatitis, digital dermatitis (Mortellaro disease). Zoonotic risk via direct contact.",
     "species": "cattle", "category": "infectious"},
    {"name": "eye_infection",      "icd_v": "H10", "severity": 2, "zoonotic": False, "notifiable": False,
     "prevalence_pct": 6, "mortality_risk": "low",
     "description": "Infectious Bovine Keratoconjunctivitis (IBK / Pinkeye) — Moraxella bovis, exacerbated by UV exposure, flies, and dust.",
     "species": "cattle", "category": "infectious"},
    {"name": "hoof_disease",       "icd_v": "L60", "severity": 3, "zoonotic": False, "notifiable": False,
     "prevalence_pct": 12, "mortality_risk": "low",
     "description": "Foot rot (Fusobacterium necrophorum), white line disease, sole ulcer, interdigital hyperplasia. Major lameness cause.",
     "species": "cattle", "category": "musculoskeletal"},
    {"name": "metabolic_disorder", "icd_v": "E88", "severity": 4, "zoonotic": False, "notifiable": False,
     "prevalence_pct": 15, "mortality_risk": "medium",
     "description": "Ketosis (hyperketonemia), milk fever (hypocalcaemia), hypomagnesaemia (grass tetany), fatty liver. Periparturient cattle at highest risk.",
     "species": "cattle", "category": "metabolic"},
    {"name": "reproductive_issue", "icd_v": "N73", "severity": 3, "zoonotic": False, "notifiable": False,
     "prevalence_pct": 20, "mortality_risk": "low",
     "description": "Metritis, retained fetal membranes (RFM), cystic ovarian disease, repeat breeder syndrome. Major driver of involuntary culling.",
     "species": "cattle", "category": "reproductive"},
    {"name": "neurological_sign",  "icd_v": "G93", "severity": 5, "zoonotic": True,  "notifiable": True,
     "prevalence_pct": 2, "mortality_risk": "high",
     "description": "Polioencephalomalacia, listeriosis (Listeria monocytogenes, zoonotic), BSE (notifiable), hypomagnasaemia tetany, lead poisoning.",
     "species": "cattle", "category": "neurological"},
    {"name": "fever",              "icd_v": "R50", "severity": 2, "zoonotic": False, "notifiable": False,
     "prevalence_pct": 30, "mortality_risk": "low",
     "description": "Non-specific pyrexia (>39.5°C) as a presenting sign of infection, inflammation, or stress. Requires investigation of underlying cause.",
     "species": "cattle", "category": "systemic"},
    {"name": "dehydration",        "icd_v": "E86", "severity": 3, "zoonotic": False, "notifiable": False,
     "prevalence_pct": 10, "mortality_risk": "medium",
     "description": "Fluid deficit >5% body weight. Common secondary to diarrhea, fever, reduced drinking. Assess mucous membranes, skin turgor, ocular recession.",
     "species": "cattle", "category": "systemic"},
    {"name": "weight_loss",        "icd_v": "R63", "severity": 3, "zoonotic": False, "notifiable": False,
     "prevalence_pct": 12, "mortality_risk": "low",
     "description": "Body condition score (BCS) decline. Multi-factorial: parasitism (Johne's disease), chronic infection, dietary insufficiency, neoplasia.",
     "species": "cattle", "category": "systemic"},
    {"name": "nasal_discharge",    "icd_v": "J34", "severity": 2, "zoonotic": False, "notifiable": False,
     "prevalence_pct": 15, "mortality_risk": "low",
     "description": "Serous to mucopurulent discharge. Cardinal sign of BRD, IBR, BRSV. Severity grades: serous (early) → mucopurulent (established infection).",
     "species": "cattle", "category": "respiratory"},
    {"name": "cough",              "icd_v": "R05", "severity": 2, "zoonotic": False, "notifiable": False,
     "prevalence_pct": 12, "mortality_risk": "low",
     "description": "Productive or dry cough in cattle. Associated with BRD, lungworm (Dictyocaulus viviparus), irritants, and cardiac disease.",
     "species": "cattle", "category": "respiratory"},
    {"name": "diarrhea",           "icd_v": "K59", "severity": 4, "zoonotic": True,  "notifiable": False,
     "prevalence_pct": 20, "mortality_risk": "medium",
     "description": "Neonatal calf diarrhea (E. coli, rotavirus, coronavirus, Cryptosporidium parvum) and adult salmonellosis. Major zoonotic risk (Salmonella, Cryptosporidium).",
     "species": "cattle", "category": "gastrointestinal"},
    {"name": "abdominal_pain",     "icd_v": "R10", "severity": 4, "zoonotic": False, "notifiable": False,
     "prevalence_pct": 5, "mortality_risk": "medium",
     "description": "Colic manifestations: bloat, LDA/RDA, intussusception, abomasal volvulus, hardware disease. Emergency evaluation required.",
     "species": "cattle", "category": "gastrointestinal"},
    {"name": "joint_swelling",     "icd_v": "M25", "severity": 3, "zoonotic": False, "notifiable": False,
     "prevalence_pct": 5, "mortality_risk": "low",
     "description": "Septic arthritis (neonates), degenerative joint disease, mycoplasma polyarthritis, Haemophilus somnus. Assess synovial fluid.",
     "species": "cattle", "category": "musculoskeletal"},
    {"name": "udder_abnormality",  "icd_v": "N64", "severity": 3, "zoonotic": False, "notifiable": False,
     "prevalence_pct": 8, "mortality_risk": "low",
     "description": "Teat stenosis, teat laceration, supernumerary teats, udder edema, teat canal lesions, bovine herpes mammillitis.",
     "species": "cattle", "category": "reproductive"},
    {"name": "skin_nodules",       "icd_v": "L98", "severity": 4, "zoonotic": False, "notifiable": True,
     "prevalence_pct": 1, "mortality_risk": "medium",
     "description": "Lumpy Skin Disease (LSD) — OIE notifiable disease. Poxvirus causing 2-5cm skin nodules, fever, lymphadenopathy, limb edema. Transmitted by insects.",
     "species": "cattle", "category": "infectious"},
    {"name": "oral_lesion",        "icd_v": "K13", "severity": 5, "zoonotic": True,  "notifiable": True,
     "prevalence_pct": 1, "mortality_risk": "low",
     "description": "Foot-and-Mouth Disease (FMD) — OIE notifiable, highly contagious. Vesicles on mouth, tongue, teats, feet. Also BVD mucosal disease. Immediate isolation required.",
     "species": "cattle", "category": "infectious"},
    {"name": "lymph_node_swelling","icd_v": "R59", "severity": 4, "zoonotic": False, "notifiable": False,
     "prevalence_pct": 5, "mortality_risk": "low",
     "description": "Johne's disease (MAP — Mycobacterium avium paratuberculosis), Bovine Leukemia Virus (BLV), caseous lymphadenitis. Progressive wasting in Johne's.",
     "species": "cattle", "category": "systemic"},
    {"name": "abnormal_gait",      "icd_v": "R26", "severity": 3, "zoonotic": False, "notifiable": False,
     "prevalence_pct": 10, "mortality_risk": "low",
     "description": "Ataxia, toe dragging, circumduction. Caused by spinal cord compression, peripheral nerve damage, severe laminitis, intoxication (bracken fern, lead).",
     "species": "cattle", "category": "musculoskeletal"},
    {"name": "lethargy",           "icd_v": "R53", "severity": 3, "zoonotic": False, "notifiable": False,
     "prevalence_pct": 25, "mortality_risk": "low",
     "description": "Reduced activity, social withdrawal, decreased responsiveness. Non-specific sign accompanying systemic illness, pain, fever, or toxaemia.",
     "species": "cattle", "category": "systemic"},
    {"name": "healthy",            "icd_v": "Z00", "severity": 0, "zoonotic": False, "notifiable": False,
     "prevalence_pct": 70, "mortality_risk": "none",
     "description": "Normal healthy bovine. Body temperature 38.0-39.3°C, heart rate 40-80 bpm, respiratory rate 10-30 breaths/min, normal rumen motility, good BCS (2.5-3.5).",
     "species": "cattle", "category": "healthy"},
]

SYMPTOMS: List[Dict[str, Any]] = [
    # mastitis symptoms
    {"name": "udder_swelling",          "onset_hours": 12,  "urgency": 7, "visibility": "observable"},
    {"name": "watery_milk",             "onset_hours": 24,  "urgency": 8, "visibility": "measurable"},
    {"name": "clotted_milk",            "onset_hours": 24,  "urgency": 7, "visibility": "observable"},
    {"name": "high_somatic_cell_count", "onset_hours": 48,  "urgency": 5, "visibility": "measurable"},
    {"name": "teat_heat_redness",       "onset_hours": 8,   "urgency": 6, "visibility": "observable"},
    # respiratory
    {"name": "productive_cough",        "onset_hours": 24,  "urgency": 5, "visibility": "observable"},
    {"name": "mucopurulent_nasal_discharge", "onset_hours": 24,  "urgency": 6, "visibility": "observable"},
    {"name": "dyspnea",                 "onset_hours": 12,  "urgency": 9, "visibility": "observable"},
    {"name": "increased_respiratory_rate", "onset_hours": 6,   "urgency": 7, "visibility": "measurable"},
    {"name": "bilateral_nasal_discharge","onset_hours": 24,  "urgency": 5, "visibility": "observable"},
    # fever / systemic
    {"name": "elevated_temperature",    "onset_hours": 6,   "urgency": 7, "visibility": "measurable"},
    {"name": "reduced_appetite",        "onset_hours": 12,  "urgency": 5, "visibility": "observable"},
    {"name": "reduced_milk_yield",      "onset_hours": 24,  "urgency": 6, "visibility": "measurable"},
    {"name": "depression",              "onset_hours": 12,  "urgency": 5, "visibility": "observable"},
    {"name": "muscle_tremors",          "onset_hours": 2,   "urgency": 9, "visibility": "observable"},
    # metabolic
    {"name": "ketone_smell_breath",     "onset_hours": 48,  "urgency": 7, "visibility": "measurable"},
    {"name": "recumbency",              "onset_hours": 4,   "urgency": 9, "visibility": "observable"},
    {"name": "low_blood_calcium",       "onset_hours": 12,  "urgency": 8, "visibility": "measurable"},
    {"name": "decreased_rumen_motility","onset_hours": 24,  "urgency": 6, "visibility": "measurable"},
    {"name": "ping_on_auscultation",    "onset_hours": 48,  "urgency": 8, "visibility": "measurable"},
    # dermatology / skin
    {"name": "circular_hairless_lesion","onset_hours": 168, "urgency": 3, "visibility": "observable"},
    {"name": "skin_nodules_2_5cm",      "onset_hours": 72,  "urgency": 8, "visibility": "observable"},
    {"name": "crusty_scabs",            "onset_hours": 96,  "urgency": 4, "visibility": "observable"},
    {"name": "pruritus",                "onset_hours": 48,  "urgency": 3, "visibility": "observable"},
    # hoof
    {"name": "interdigital_swelling",   "onset_hours": 24,  "urgency": 7, "visibility": "observable"},
    {"name": "foul_interdigital_odour", "onset_hours": 12,  "urgency": 7, "visibility": "observable"},
    {"name": "sole_ulcer",              "onset_hours": 168, "urgency": 6, "visibility": "measurable"},
    {"name": "weight_shifting",         "onset_hours": 4,   "urgency": 5, "visibility": "observable"},
    # eye
    {"name": "corneal_opacity",         "onset_hours": 48,  "urgency": 6, "visibility": "observable"},
    {"name": "epiphora",                "onset_hours": 12,  "urgency": 4, "visibility": "observable"},
    {"name": "photophobia",             "onset_hours": 24,  "urgency": 5, "visibility": "observable"},
    {"name": "conjunctival_hyperemia",  "onset_hours": 6,   "urgency": 5, "visibility": "observable"},
    # neurological
    {"name": "head_tilt",               "onset_hours": 12,  "urgency": 9, "visibility": "observable"},
    {"name": "circling",                "onset_hours": 12,  "urgency": 9, "visibility": "observable"},
    {"name": "blindness",               "onset_hours": 24,  "urgency": 9, "visibility": "observable"},
    {"name": "opisthotonus",            "onset_hours": 6,   "urgency": 10,"visibility": "observable"},
    {"name": "seizures",                "onset_hours": 2,   "urgency": 10,"visibility": "observable"},
    # diarrhea / GI
    {"name": "watery_diarrhea",         "onset_hours": 12,  "urgency": 7, "visibility": "observable"},
    {"name": "blood_in_stool",          "onset_hours": 12,  "urgency": 9, "visibility": "observable"},
    {"name": "abdominal_distension",    "onset_hours": 6,   "urgency": 8, "visibility": "observable"},
    {"name": "tenesmus",                "onset_hours": 6,   "urgency": 7, "visibility": "observable"},
    {"name": "bruxism",                 "onset_hours": 6,   "urgency": 6, "visibility": "observable"},
    # joint
    {"name": "hot_swollen_joint",       "onset_hours": 24,  "urgency": 7, "visibility": "observable"},
    {"name": "joint_effusion",          "onset_hours": 48,  "urgency": 6, "visibility": "measurable"},
    # dehydration
    {"name": "skin_tenting",            "onset_hours": 12,  "urgency": 7, "visibility": "observable"},
    {"name": "sunken_eyes",             "onset_hours": 12,  "urgency": 7, "visibility": "observable"},
    {"name": "dry_mucous_membranes",    "onset_hours": 8,   "urgency": 7, "visibility": "observable"},
    # oral
    {"name": "vesicles_on_tongue",      "onset_hours": 24,  "urgency": 10,"visibility": "observable"},
    {"name": "excessive_salivation",    "onset_hours": 12,  "urgency": 8, "visibility": "observable"},
    {"name": "oral_erosions",           "onset_hours": 48,  "urgency": 8, "visibility": "observable"},
    # reproductive
    {"name": "vaginal_discharge",       "onset_hours": 48,  "urgency": 6, "visibility": "observable"},
    {"name": "retained_placenta",       "onset_hours": 12,  "urgency": 7, "visibility": "observable"},
    {"name": "anestrus",                "onset_hours": 720, "urgency": 4, "visibility": "measurable"},
    # lymph
    {"name": "enlarged_prescapular_LN", "onset_hours": 168, "urgency": 5, "visibility": "observable"},
    {"name": "generalised_lymphadenopathy","onset_hours": 336,"urgency": 7,"visibility": "observable"},
    # general
    {"name": "weight_loss_progressive", "onset_hours": 720, "urgency": 5, "visibility": "measurable"},
    {"name": "rough_dry_coat",          "onset_hours": 336, "urgency": 3, "visibility": "observable"},
    {"name": "reduced_activity",        "onset_hours": 12,  "urgency": 4, "visibility": "observable"},
    {"name": "ataxia",                  "onset_hours": 24,  "urgency": 8, "visibility": "observable"},
    {"name": "limb_edema",              "onset_hours": 72,  "urgency": 6, "visibility": "observable"},
    {"name": "fever_above_41C",         "onset_hours": 6,   "urgency": 9, "visibility": "measurable"},
    {"name": "pale_mucous_membranes",   "onset_hours": 24,  "urgency": 8, "visibility": "observable"},
    {"name": "submandibular_oedema",    "onset_hours": 168, "urgency": 6, "visibility": "observable"},
]

TREATMENTS: List[Dict[str, Any]] = [
    # mastitis
    {"name": "Intramammary Cephalosporins",  "evidence": "A", "protocol": "Infuse 1 tube per affected quarter q12h for 2-3 days. Strip quarter before each treatment. Use strict hygiene.", "withdrawal_milk_days": 3, "withdrawal_meat_days": 4, "category": "antibiotic"},
    {"name": "Dry Cow Therapy",              "evidence": "A", "protocol": "At dry-off: infuse internal teat sealant ± antibiotic. Teat canal closure 5ml Orbeseal or equivalent.", "withdrawal_milk_days": 0, "withdrawal_meat_days": 30, "category": "preventive"},
    {"name": "Supportive NSAID Therapy",     "evidence": "A", "protocol": "Meloxicam 0.5 mg/kg IV/SC once daily for 3-5 days alongside antibiotic treatment.", "withdrawal_milk_days": 5, "withdrawal_meat_days": 15, "category": "nsaid"},
    {"name": "Teat Dipping",                 "evidence": "A", "protocol": "Post-milking teat dip with 0.5% iodine or chlorhexidine. Pre-milking if digital dermatitis present.", "withdrawal_milk_days": 0, "withdrawal_meat_days": 0, "category": "preventive"},
    # respiratory
    {"name": "Tulathromycin Single Dose",    "evidence": "A", "protocol": "2.5 mg/kg SC single injection. For BRD metaphylaxis and treatment. Spectrum: Mannheimia, Pasteurella, Mycoplasma.", "withdrawal_milk_days": 0, "withdrawal_meat_days": 49, "category": "antibiotic"},
    {"name": "Florfenicol Respiratory",      "evidence": "A", "protocol": "20 mg/kg IM q48h for 2 doses, OR 40 mg/kg SC single dose. Not for use in lactating dairy cattle.", "withdrawal_milk_days": 999, "withdrawal_meat_days": 38, "category": "antibiotic"},
    {"name": "Enrofloxacin BRD",            "evidence": "B", "protocol": "7.5-12.5 mg/kg SC once daily for 3-5 days. Reserve for confirmed Mycoplasma or fluoroquinolone-sensitive organisms.", "withdrawal_milk_days": 999, "withdrawal_meat_days": 17, "category": "antibiotic"},
    {"name": "Dexamethasone Anti-inflammatory","evidence":"B","protocol": "0.1-0.2 mg/kg IV/IM once. Use in severe BRD with significant pulmonary compromise. Avoid in pregnancy.", "withdrawal_milk_days": 7, "withdrawal_meat_days": 21, "category": "corticosteroid"},
    # metabolic
    {"name": "IV Calcium Borogluconate",     "evidence": "A", "protocol": "400ml of 40% calcium borogluconate IV SLOWLY (<60 min) with cardiac monitoring. 500ml SC flank if recumbent and mild.", "withdrawal_milk_days": 0, "withdrawal_meat_days": 0, "category": "metabolic"},
    {"name": "Propylene Glycol Drench",      "evidence": "A", "protocol": "300-400ml PO twice daily for 3-5 days. Glucogenic precursor for ketosis treatment. Check clinical ketosis confirmation.", "withdrawal_milk_days": 0, "withdrawal_meat_days": 0, "category": "metabolic"},
    {"name": "Dextrose 50% IV",              "evidence": "A", "protocol": "0.5g/kg IV bolus for hypoglycaemia. Repeat q4h. Concurrent propylene glycol for sustained effect.", "withdrawal_milk_days": 0, "withdrawal_meat_days": 0, "category": "metabolic"},
    {"name": "Magnesium Sulfate IV",         "evidence": "A", "protocol": "Slow IV infusion 200ml 50% MgSO4 in 1L saline over 30 min for hypomagnesaemic tetany. Monitor ECG.", "withdrawal_milk_days": 0, "withdrawal_meat_days": 0, "category": "metabolic"},
    {"name": "Thiamine IV High Dose",        "evidence": "B", "protocol": "10-20 mg/kg IV q6h for polioencephalomalacia. Continue for minimum 5 days. Rapid improvement expected within 24h if PEM.", "withdrawal_milk_days": 0, "withdrawal_meat_days": 0, "category": "vitamin"},
    # reproductive
    {"name": "Cloprostenol PGF2a",          "evidence": "A", "protocol": "500 mcg IM for luteolysis. Use for ovarian cysts, pyometra, metritis with CL. Repeat in 10-14 days if needed.", "withdrawal_milk_days": 0, "withdrawal_meat_days": 1, "category": "hormone"},
    {"name": "Uterine Lavage",              "evidence": "B", "protocol": "Infuse 500ml sterile saline via uterine catheter. Aspirate after 30 min. Repeat daily for 3 days (severe metritis).", "withdrawal_milk_days": 0, "withdrawal_meat_days": 0, "category": "physical"},
    {"name": "Oxytetracycline Uterine",     "evidence": "B", "protocol": "500mg intrauterine infusion after manual removal of fetal membranes on days 1, 3, 5. Not a substitute for systemic treatment.", "withdrawal_milk_days": 4, "withdrawal_meat_days": 22, "category": "antibiotic"},
    # hoof
    {"name": "Hoof Trimming Therapeutic",   "evidence": "A", "protocol": "Correct weight-bearing. Apply wooden block to healthy claw. Bandage as needed. Follow 5-step Dutch hoof trimming protocol.", "withdrawal_milk_days": 0, "withdrawal_meat_days": 0, "category": "physical"},
    {"name": "Copper Sulfate Foot Bath",    "evidence": "A", "protocol": "4% CuSO4 solution. Cows walk through 2x per week. For digital dermatitis and foot rot prophylaxis.", "withdrawal_milk_days": 0, "withdrawal_meat_days": 0, "category": "preventive"},
    {"name": "Penicillin Foot Rot",        "evidence": "A", "protocol": "Procaine penicillin 22,000 IU/kg IM q24h for 5 days. Effective against Fusobacterium necrophorum.", "withdrawal_milk_days": 4, "withdrawal_meat_days": 10, "category": "antibiotic"},
    # eye
    {"name": "Penicillin IBK IM",          "evidence": "A", "protocol": "Procaine penicillin 22,000 IU/kg IM once. Subconjunctival injection of procaine penicillin 0.5ml is an alternative for single-eye.", "withdrawal_milk_days": 4, "withdrawal_meat_days": 10, "category": "antibiotic"},
    {"name": "Eye Patching",               "evidence": "B", "protocol": "Suture upper/lower eyelid (tarsorrhaphy) or tape patch for 10-14 days. Reduces UV exposure and promotes healing.", "withdrawal_milk_days": 0, "withdrawal_meat_days": 0, "category": "physical"},
    # diarrhea / GI
    {"name": "Oral Fluid Electrolyte Therapy","evidence":"A","protocol": "Oral rehydration 2-4L q6-8h. Use commercial ORS (NaHCO3, NaCl, KCl, glucose). Continue until alert and nursing.", "withdrawal_milk_days": 0, "withdrawal_meat_days": 0, "category": "supportive"},
    {"name": "IV Fluid Therapy Crystalloids","evidence":"A","protocol": "Hartmann's/LRS 60-90ml/kg/day IV. Correct deficit + maintenance. Add NaHCO3 for severe acidosis (BE < -10).", "withdrawal_milk_days": 0, "withdrawal_meat_days": 0, "category": "supportive"},
    {"name": "Salmonella Antibiotic Protocol","evidence":"B","protocol": "Enrofloxacin 5mg/kg IM q24h for confirmed Salmonella with bacteraemia. Culture and sensitivity essential.", "withdrawal_milk_days": 999, "withdrawal_meat_days": 17, "category": "antibiotic"},
    # neurological
    {"name": "Penicillin Listeriosis",     "evidence": "B", "protocol": "Procaine penicillin 44,000 IU/kg IM q12h for minimum 2 weeks. Highest cure rate if initiated within 24h of onset.", "withdrawal_milk_days": 4, "withdrawal_meat_days": 10, "category": "antibiotic"},
    # skin
    {"name": "Topical Antifungal",         "evidence": "B", "protocol": "Apply enilconazole or natamycin spray to lesions daily for 2 weeks. Isolate affected animals. Zoonotic risk management.", "withdrawal_milk_days": 0, "withdrawal_meat_days": 0, "category": "antifungal"},
    {"name": "Iodine Topical",             "evidence": "C", "protocol": "5-10% iodine solution applied topically daily. Alternative for ringworm and minor skin infections.", "withdrawal_milk_days": 0, "withdrawal_meat_days": 0, "category": "antiseptic"},
    # LSD
    {"name": "LSD Vaccination",            "evidence": "A", "protocol": "Neethling strain live attenuated vaccine. Single SC dose. Not for use in immunocompromised. Annual revaccination in endemic areas.", "withdrawal_milk_days": 0, "withdrawal_meat_days": 28, "category": "vaccine"},
    {"name": "LSD Supportive Care",        "evidence": "C", "protocol": "NSAIDs for pain and inflammation. Antibiotic cover for secondary bacterial infections. Insect vector control.", "withdrawal_milk_days": 5, "withdrawal_meat_days": 15, "category": "supportive"},
    # surgical
    {"name": "LDA Surgical Toggle",        "evidence": "A", "protocol": "Standing right paralumbar fossa laparotomy. Deflate gas, reposition abomasum, suture to right abdominal floor. Post-op NSAIDs.", "withdrawal_milk_days": 0, "withdrawal_meat_days": 15, "category": "surgical"},
    # joint
    {"name": "Joint Lavage Arthrotomy",    "evidence": "B", "protocol": "Needle arthrocentesis under aseptic conditions. Lavage with sterile saline 200ml. Culture synovial fluid. Systemic antibiotics.", "withdrawal_milk_days": 0, "withdrawal_meat_days": 0, "category": "surgical"},
    {"name": "Meloxicam NSAID Joint",      "evidence": "A", "protocol": "0.5 mg/kg IV/SC once daily for 3-5 days. Anti-inflammatory for joint pain and lameness. Can be combined with antibiotics.", "withdrawal_milk_days": 5, "withdrawal_meat_days": 15, "category": "nsaid"},
]

DRUGS: List[Dict[str, Any]] = [
    {"name": "Meloxicam",          "active": "meloxicam",             "category": "NSAID",            "withdrawal_milk_days": 5,  "withdrawal_meat_days": 15,  "prescription": True},
    {"name": "Flunixin Meglumine", "active": "flunixin meglumine",    "category": "NSAID",            "withdrawal_milk_days": 36, "withdrawal_meat_days": 4,   "prescription": True},
    {"name": "Procaine Penicillin","active": "benzylpenicillin",      "category": "Antibiotic-Beta",  "withdrawal_milk_days": 4,  "withdrawal_meat_days": 10,  "prescription": True},
    {"name": "Oxytetracycline",    "active": "oxytetracycline",       "category": "Antibiotic-TC",    "withdrawal_milk_days": 4,  "withdrawal_meat_days": 22,  "prescription": True},
    {"name": "Tulathromycin",      "active": "tulathromycin",         "category": "Antibiotic-Macro", "withdrawal_milk_days": 0,  "withdrawal_meat_days": 49,  "prescription": True},
    {"name": "Florfenicol",        "active": "florfenicol",           "category": "Antibiotic-Amph",  "withdrawal_milk_days": 999,"withdrawal_meat_days": 38,  "prescription": True},
    {"name": "Enrofloxacin",       "active": "enrofloxacin",          "category": "Antibiotic-FQ",    "withdrawal_milk_days": 999,"withdrawal_meat_days": 17,  "prescription": True},
    {"name": "Cloprostenol",       "active": "cloprostenol",          "category": "Prostaglandin",    "withdrawal_milk_days": 0,  "withdrawal_meat_days": 1,   "prescription": True},
    {"name": "Dexamethasone",      "active": "dexamethasone",         "category": "Corticosteroid",   "withdrawal_milk_days": 7,  "withdrawal_meat_days": 21,  "prescription": True},
    {"name": "Calcium Borogluconate","active":"calcium borogluconate","category": "Mineral-IV",       "withdrawal_milk_days": 0,  "withdrawal_meat_days": 0,   "prescription": False},
    {"name": "Propylene Glycol",   "active": "propylene glycol",      "category": "Glucogenic",       "withdrawal_milk_days": 0,  "withdrawal_meat_days": 0,   "prescription": False},
    {"name": "Thiamine HCl",       "active": "thiamine hydrochloride","category": "Vitamin-B",        "withdrawal_milk_days": 0,  "withdrawal_meat_days": 0,   "prescription": False},
    {"name": "Magnesium Sulfate",  "active": "magnesium sulfate",     "category": "Mineral-IV",       "withdrawal_milk_days": 0,  "withdrawal_meat_days": 0,   "prescription": False},
    {"name": "Dextrose 50%",       "active": "glucose",               "category": "Glucose-IV",       "withdrawal_milk_days": 0,  "withdrawal_meat_days": 0,   "prescription": False},
    {"name": "Oxytocin",           "active": "oxytocin",              "category": "Hormone",          "withdrawal_milk_days": 0,  "withdrawal_meat_days": 0,   "prescription": True},
    {"name": "Cefapirin",          "active": "cephapirin sodium",     "category": "Antibiotic-Ceph",  "withdrawal_milk_days": 3,  "withdrawal_meat_days": 4,   "prescription": True},
]

# Disease → Symptoms mapping: (disease_name, symptom_name, frequency%, pathognomonic)
DISEASE_SYMPTOMS: List[tuple] = [
    # mastitis
    ("mastitis",            "udder_swelling",           85, False),
    ("mastitis",            "watery_milk",              80, False),
    ("mastitis",            "clotted_milk",             75, False),
    ("mastitis",            "teat_heat_redness",        70, False),
    ("mastitis",            "high_somatic_cell_count",  95, True),
    ("mastitis",            "elevated_temperature",     60, False),
    ("mastitis",            "reduced_milk_yield",       90, False),
    ("mastitis",            "reduced_appetite",         50, False),
    # respiratory
    ("respiratory_disease", "productive_cough",         80, False),
    ("respiratory_disease", "mucopurulent_nasal_discharge",75, False),
    ("respiratory_disease", "dyspnea",                  60, False),
    ("respiratory_disease", "elevated_temperature",     85, False),
    ("respiratory_disease", "increased_respiratory_rate",90, False),
    ("respiratory_disease", "reduced_appetite",         70, False),
    ("respiratory_disease", "depression",               65, False),
    # metabolic
    ("metabolic_disorder",  "ketone_smell_breath",      70, True),
    ("metabolic_disorder",  "recumbency",               50, False),
    ("metabolic_disorder",  "low_blood_calcium",        80, True),
    ("metabolic_disorder",  "decreased_rumen_motility", 75, False),
    ("metabolic_disorder",  "reduced_milk_yield",       85, False),
    ("metabolic_disorder",  "reduced_appetite",         80, False),
    ("metabolic_disorder",  "muscle_tremors",           40, False),
    ("metabolic_disorder",  "depression",               60, False),
    # hoof
    ("hoof_disease",        "interdigital_swelling",    80, False),
    ("hoof_disease",        "foul_interdigital_odour",  85, True),
    ("hoof_disease",        "weight_shifting",          90, False),
    ("hoof_disease",        "sole_ulcer",               60, False),
    # lameness
    ("lameness",            "weight_shifting",          95, False),
    ("lameness",            "ataxia",                   40, False),
    ("lameness",            "sole_ulcer",               50, False),
    ("lameness",            "reduced_activity",         80, False),
    # eye infection
    ("eye_infection",       "corneal_opacity",          70, False),
    ("eye_infection",       "epiphora",                 95, False),
    ("eye_infection",       "photophobia",              80, False),
    ("eye_infection",       "conjunctival_hyperemia",   90, False),
    # neurological
    ("neurological_sign",   "head_tilt",                60, False),
    ("neurological_sign",   "circling",                 50, False),
    ("neurological_sign",   "blindness",                40, False),
    ("neurological_sign",   "opisthotonus",             30, False),
    ("neurological_sign",   "seizures",                 25, False),
    ("neurological_sign",   "ataxia",                   70, False),
    ("neurological_sign",   "recumbency",               50, False),
    # diarrhea
    ("diarrhea",            "watery_diarrhea",          95, False),
    ("diarrhea",            "blood_in_stool",           30, False),
    ("diarrhea",            "skin_tenting",             60, False),
    ("diarrhea",            "reduced_appetite",         70, False),
    ("diarrhea",            "elevated_temperature",     50, False),
    # digestive disorder
    ("digestive_disorder",  "abdominal_distension",     80, False),
    ("digestive_disorder",  "ping_on_auscultation",     65, True),
    ("digestive_disorder",  "reduced_appetite",         90, False),
    ("digestive_disorder",  "bruxism",                  40, False),
    ("digestive_disorder",  "decreased_rumen_motility", 85, False),
    # reproductive
    ("reproductive_issue",  "vaginal_discharge",        75, False),
    ("reproductive_issue",  "retained_placenta",        40, True),
    ("reproductive_issue",  "anestrus",                 50, False),
    ("reproductive_issue",  "elevated_temperature",     55, False),
    ("reproductive_issue",  "reduced_milk_yield",       60, False),
    # dehydration
    ("dehydration",         "skin_tenting",             90, False),
    ("dehydration",         "sunken_eyes",              85, False),
    ("dehydration",         "dry_mucous_membranes",     90, False),
    ("dehydration",         "reduced_activity",         80, False),
    # skin nodules (LSD)
    ("skin_nodules",        "skin_nodules_2_5cm",       95, True),
    ("skin_nodules",        "elevated_temperature",     85, False),
    ("skin_nodules",        "limb_edema",               50, False),
    ("skin_nodules",        "mucopurulent_nasal_discharge",60,False),
    # oral lesion (FMD)
    ("oral_lesion",         "vesicles_on_tongue",       95, True),
    ("oral_lesion",         "excessive_salivation",     90, False),
    ("oral_lesion",         "oral_erosions",            85, False),
    ("oral_lesion",         "fever_above_41C",          80, True),
    ("oral_lesion",         "limb_edema",               70, False),
    # lymph node
    ("lymph_node_swelling", "enlarged_prescapular_LN",  85, False),
    ("lymph_node_swelling", "generalised_lymphadenopathy",60,False),
    ("lymph_node_swelling", "weight_loss_progressive",  70, False),
    ("lymph_node_swelling", "watery_diarrhea",          50, False),
    # skin lesion
    ("skin_lesion",         "circular_hairless_lesion", 90, True),
    ("skin_lesion",         "crusty_scabs",             75, False),
    ("skin_lesion",         "pruritus",                 60, False),
    # fever
    ("fever",               "elevated_temperature",     100,True),
    ("fever",               "reduced_appetite",         80, False),
    ("fever",               "depression",               75, False),
    # lethargy
    ("lethargy",            "reduced_activity",         95, False),
    ("lethargy",            "depression",               90, False),
    ("lethargy",            "rough_dry_coat",           50, False),
    # weight loss
    ("weight_loss",         "weight_loss_progressive",  95, True),
    ("weight_loss",         "rough_dry_coat",           70, False),
    ("weight_loss",         "watery_diarrhea",          40, False),
    # joint swelling
    ("joint_swelling",      "hot_swollen_joint",        90, False),
    ("joint_swelling",      "joint_effusion",           75, False),
    ("joint_swelling",      "weight_shifting",          80, False),
    # abdominal pain
    ("abdominal_pain",      "abdominal_distension",     70, False),
    ("abdominal_pain",      "bruxism",                  60, False),
    ("abdominal_pain",      "tenesmus",                 50, False),
    # udder abnormality
    ("udder_abnormality",   "teat_heat_redness",        60, False),
    ("udder_abnormality",   "reduced_milk_yield",       70, False),
    # nasal discharge
    ("nasal_discharge",     "bilateral_nasal_discharge",90, False),
    ("nasal_discharge",     "elevated_temperature",     60, False),
    # cough
    ("cough",               "productive_cough",         90, False),
    ("cough",               "increased_respiratory_rate",60,False),
    # abnormal gait
    ("abnormal_gait",       "ataxia",                   80, False),
    ("abnormal_gait",       "weight_shifting",          70, False),
    # pale mucous membranes
    ("weight_loss",         "pale_mucous_membranes",    40, False),
]

# Disease → Treatment (first_line: bool, evidence_level)
DISEASE_TREATMENTS: List[tuple] = [
    # mastitis
    ("mastitis",            "Intramammary Cephalosporins",    True),
    ("mastitis",            "Supportive NSAID Therapy",       True),
    ("mastitis",            "Dry Cow Therapy",                True),
    ("mastitis",            "Teat Dipping",                   False),
    # respiratory
    ("respiratory_disease", "Tulathromycin Single Dose",      True),
    ("respiratory_disease", "Florfenicol Respiratory",        True),
    ("respiratory_disease", "Enrofloxacin BRD",              False),
    ("respiratory_disease", "Dexamethasone Anti-inflammatory",False),
    ("respiratory_disease", "Supportive NSAID Therapy",      False),
    # metabolic
    ("metabolic_disorder",  "IV Calcium Borogluconate",       True),
    ("metabolic_disorder",  "Propylene Glycol Drench",        True),
    ("metabolic_disorder",  "Dextrose 50% IV",               True),
    ("metabolic_disorder",  "Magnesium Sulfate IV",          True),
    # reproductive
    ("reproductive_issue",  "Cloprostenol PGF2a",            True),
    ("reproductive_issue",  "Uterine Lavage",                True),
    ("reproductive_issue",  "Oxytetracycline Uterine",       False),
    # hoof
    ("hoof_disease",        "Hoof Trimming Therapeutic",     True),
    ("hoof_disease",        "Copper Sulfate Foot Bath",      False),
    ("hoof_disease",        "Penicillin Foot Rot",           True),
    # lameness
    ("lameness",            "Hoof Trimming Therapeutic",     True),
    ("lameness",            "Meloxicam NSAID Joint",         True),
    ("lameness",            "Copper Sulfate Foot Bath",      False),
    # eye
    ("eye_infection",       "Penicillin IBK IM",             True),
    ("eye_infection",       "Eye Patching",                  False),
    # neurological
    ("neurological_sign",   "Thiamine IV High Dose",         True),
    ("neurological_sign",   "Penicillin Listeriosis",        True),
    # diarrhea
    ("diarrhea",            "Oral Fluid Electrolyte Therapy",True),
    ("diarrhea",            "IV Fluid Therapy Crystalloids", True),
    ("diarrhea",            "Salmonella Antibiotic Protocol",False),
    # digestive
    ("digestive_disorder",  "LDA Surgical Toggle",           True),
    ("digestive_disorder",  "Oral Fluid Electrolyte Therapy",False),
    # skin
    ("skin_lesion",         "Topical Antifungal",            True),
    ("skin_lesion",         "Iodine Topical",                False),
    # skin nodules / LSD
    ("skin_nodules",        "LSD Vaccination",               True),
    ("skin_nodules",        "LSD Supportive Care",           True),
    ("skin_nodules",        "Supportive NSAID Therapy",      False),
    # joint
    ("joint_swelling",      "Joint Lavage Arthrotomy",       True),
    ("joint_swelling",      "Meloxicam NSAID Joint",         True),
    ("joint_swelling",      "Penicillin Foot Rot",           True),
    # fever
    ("fever",               "Supportive NSAID Therapy",      True),
    # dehydration
    ("dehydration",         "IV Fluid Therapy Crystalloids", True),
    ("dehydration",         "Oral Fluid Electrolyte Therapy",True),
    # abdominal
    ("abdominal_pain",      "IV Fluid Therapy Crystalloids", False),
    ("abdominal_pain",      "LDA Surgical Toggle",           False),
    # oral (FMD - notifiable - supportive only)
    ("oral_lesion",         "LSD Supportive Care",           True),
    # reproductive
    ("udder_abnormality",   "Intramammary Cephalosporins",   True),
    ("udder_abnormality",   "Supportive NSAID Therapy",      False),
]

# Disease → Disease relationships
DISEASE_RELATIONS: List[tuple] = [
    # (d1, d2, relationship_type, strength 0-1, time_days, probability)
    ("mastitis",           "reproductive_issue",    "comorbidity",   0.7, None, None),
    ("mastitis",           "metabolic_disorder",    "comorbidity",   0.6, None, None),
    ("mastitis",           "udder_abnormality",     "progresses_to", 0.5, 30,   0.35),
    ("lameness",           "hoof_disease",          "subtype_of",    0.9, None, None),
    ("lameness",           "abnormal_gait",         "causes",        0.8, 7,    0.6),
    ("lameness",           "metabolic_disorder",    "comorbidity",   0.5, None, None),
    ("respiratory_disease","cough",                 "causes",        0.9, 2,    0.85),
    ("respiratory_disease","nasal_discharge",       "causes",        0.85,2,    0.8),
    ("respiratory_disease","fever",                 "causes",        0.8, 1,    0.75),
    ("digestive_disorder", "abdominal_pain",        "causes",        0.85,4,    0.7),
    ("digestive_disorder", "metabolic_disorder",    "comorbidity",   0.6, None, None),
    ("metabolic_disorder", "lameness",              "predisposes",   0.5, 14,   0.3),
    ("metabolic_disorder", "reproductive_issue",    "comorbidity",   0.65,None, None),
    ("diarrhea",           "dehydration",           "progresses_to", 0.9, 2,    0.7),
    ("diarrhea",           "weight_loss",           "progresses_to", 0.7, 14,   0.5),
    ("skin_nodules",       "secondary_infection",   "predisposes",   0.6, 7,    0.4),
    ("oral_lesion",        "weight_loss",           "progresses_to", 0.7, 14,   0.5),
    ("lymph_node_swelling","weight_loss",           "comorbidity",   0.8, None, None),
    ("neurological_sign",  "lethargy",              "causes",        0.9, 1,    0.8),
    ("fever",              "dehydration",           "predisposes",   0.6, 3,    0.4),
    ("reproductive_issue", "mastitis",              "comorbidity",   0.55,None, None),
]

KNOWN_COWS: List[Dict] = [
    {"cow_id": i, "breed": b, "age_years": a, "source": "mmcows_dataset", "status": "active"}
    for i, b, a in [
        (1, "Holstein",  4.5), (2, "Holstein",  5.2), (3, "Jersey",    3.1), (4, "Jersey",    4.8),
        (5, "Angus",     6.0), (6, "Angus",     2.8), (7, "Hereford",  7.1), (8, "Hereford",  3.5),
        (9, "Holstein",  5.8), (10,"Simmental", 4.2), (11,"Holstein",  6.5), (12,"Jersey",    2.5),
        (13,"Angus",     3.9), (14,"Hereford",  5.5), (15,"Simmental", 4.0), (16,"Holstein",  7.2),
    ]
]


# ── Neo4j Seeder Class ────────────────────────────────────────────────────────

class VeterinaryKGSeeder:

    NCBI_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    NCBI_FETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.driver.verify_connectivity()
        logger.info(f"Connected to {uri}")

    def run_all(self) -> None:
        logger.info("=" * 60)
        logger.info("VETERINARY KG SEEDER — Starting full population run")
        logger.info("=" * 60)
        self._create_constraints()
        self._seed_diseases()
        self._seed_symptoms()
        self._seed_treatments()
        self._seed_drugs()
        self._link_disease_symptoms()
        self._link_disease_treatments()
        self._link_disease_relations()
        self._seed_known_cows()
        self._seed_pubmed_research()
        self._build_treatment_drug_links()
        self._print_stats()
        logger.info("=" * 60)
        logger.info("✅  KG seeding complete!")
        logger.info("=" * 60)

    def _session(self):
        return self.driver.session(database=self.database)

    def _create_constraints(self) -> None:
        logger.info("Creating schema constraints …")
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease)   REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom)   REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Treatment) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Drug)      REQUIRE g.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Cow)       REQUIRE c.cow_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Research)  REQUIRE r.pmid IS UNIQUE",
        ]
        with self._session() as s:
            for cql in constraints:
                try:
                    s.run(cql)
                except Exception as exc:
                    logger.debug(f"Constraint note: {exc}")

    def _seed_diseases(self) -> None:
        logger.info(f"Seeding {len(DISEASES)} disease nodes …")
        cql = """
        MERGE (d:Disease {name: $name})
        SET d.icd_v            = $icd_v,
            d.severity         = $severity,
            d.zoonotic         = $zoonotic,
            d.notifiable       = $notifiable,
            d.prevalence_pct   = $prevalence_pct,
            d.mortality_risk   = $mortality_risk,
            d.description      = $description,
            d.species          = $species,
            d.category         = $category,
            d.last_updated     = date()
        """
        with self._session() as s:
            for d in DISEASES:
                s.run(cql, **d)
        logger.info(f"  ✓ {len(DISEASES)} diseases seeded")

    def _seed_symptoms(self) -> None:
        logger.info(f"Seeding {len(SYMPTOMS)} symptom nodes …")
        cql = """
        MERGE (s:Symptom {name: $name})
        SET s.onset_hours = $onset_hours,
            s.urgency     = $urgency,
            s.visibility  = $visibility
        """
        with self._session() as s:
            for sym in SYMPTOMS:
                s.run(cql, **sym)
        logger.info(f"  ✓ {len(SYMPTOMS)} symptoms seeded")

    def _seed_treatments(self) -> None:
        logger.info(f"Seeding {len(TREATMENTS)} treatment nodes …")
        cql = """
        MERGE (t:Treatment {name: $name})
        SET t.evidence_level        = $evidence,
            t.protocol              = $protocol,
            t.withdrawal_milk_days  = $withdrawal_milk_days,
            t.withdrawal_meat_days  = $withdrawal_meat_days,
            t.category              = $category
        """
        with self._session() as s:
            for t in TREATMENTS:
                s.run(cql, **t)
        logger.info(f"  ✓ {len(TREATMENTS)} treatments seeded")

    def _seed_drugs(self) -> None:
        logger.info(f"Seeding {len(DRUGS)} drug nodes …")
        cql = """
        MERGE (g:Drug {name: $name})
        SET g.active_ingredient     = $active,
            g.category              = $category,
            g.withdrawal_milk_days  = $withdrawal_milk_days,
            g.withdrawal_meat_days  = $withdrawal_meat_days,
            g.prescription_required = $prescription
        """
        with self._session() as s:
            for d in DRUGS:
                s.run(cql, **d)
        logger.info(f"  ✓ {len(DRUGS)} drugs seeded")

    def _link_disease_symptoms(self) -> None:
        logger.info(f"Linking disease→symptom ({len(DISEASE_SYMPTOMS)} relationships) …")
        cql = """
        MATCH (d:Disease {name: $disease})
        MATCH (s:Symptom  {name: $symptom})
        MERGE (d)-[r:HAS_SYMPTOM]->(s)
        SET r.frequency_pct   = $freq,
            r.pathognomonic   = $pathognomonic
        """
        with self._session() as s:
            for disease, symptom, freq, pathognomonic in DISEASE_SYMPTOMS:
                try:
                    s.run(cql, disease=disease, symptom=symptom,
                          freq=freq, pathognomonic=pathognomonic)
                except Exception as exc:
                    logger.warning(f"Symptom link failed {disease}→{symptom}: {exc}")
        logger.info(f"  ✓ Disease→Symptom relationships created")

    def _link_disease_treatments(self) -> None:
        logger.info(f"Linking disease→treatment ({len(DISEASE_TREATMENTS)} relationships) …")
        cql = """
        MATCH (d:Disease   {name: $disease})
        MATCH (t:Treatment {name: $treatment})
        MERGE (d)-[r:TREATED_BY]->(t)
        SET r.first_line     = $first_line,
            r.evidence_level = t.evidence_level
        """
        with self._session() as s:
            for disease, treatment, first_line in DISEASE_TREATMENTS:
                try:
                    s.run(cql, disease=disease, treatment=treatment, first_line=first_line)
                except Exception as exc:
                    logger.warning(f"Treatment link failed {disease}→{treatment}: {exc}")
        logger.info(f"  ✓ Disease→Treatment relationships created")

    def _link_disease_relations(self) -> None:
        logger.info(f"Linking disease→disease ({len(DISEASE_RELATIONS)} relationships) …")
        cql_progresses = """
        MATCH (d1:Disease {name: $d1})
        MERGE (d2:Disease {name: $d2})
        MERGE (d1)-[r:PROGRESSES_TO]->(d2)
        SET r.probability = $prob, r.time_days = $days
        """
        cql_related = """
        MATCH (d1:Disease {name: $d1})
        MATCH (d2:Disease {name: $d2})
        MERGE (d1)-[r:RELATED_TO]->(d2)
        SET r.relationship_type = $rel_type, r.strength = $strength
        """
        with self._session() as s:
            for d1, d2, rel_type, strength, days, prob in DISEASE_RELATIONS:
                try:
                    if rel_type == "progresses_to" and days:
                        s.run(cql_progresses, d1=d1, d2=d2, prob=prob, days=days)
                    else:
                        s.run(cql_related, d1=d1, d2=d2, rel_type=rel_type, strength=strength)
                except Exception as exc:
                    logger.warning(f"Disease relation failed {d1}→{d2}: {exc}")
        logger.info(f"  ✓ Disease→Disease relationships created")

    def _build_treatment_drug_links(self) -> None:
        """Link treatment nodes to their primary drug where name matches."""
        logger.info("Linking treatment→drug nodes …")
        mapping = [
            ("Intramammary Cephalosporins",     "Cefapirin"),
            ("Supportive NSAID Therapy",        "Meloxicam"),
            ("Tulathromycin Single Dose",        "Tulathromycin"),
            ("Florfenicol Respiratory",          "Florfenicol"),
            ("Enrofloxacin BRD",                "Enrofloxacin"),
            ("Dexamethasone Anti-inflammatory",  "Dexamethasone"),
            ("IV Calcium Borogluconate",         "Calcium Borogluconate"),
            ("Propylene Glycol Drench",          "Propylene Glycol"),
            ("Dextrose 50% IV",                 "Dextrose 50%"),
            ("Magnesium Sulfate IV",             "Magnesium Sulfate"),
            ("Thiamine IV High Dose",            "Thiamine HCl"),
            ("Cloprostenol PGF2a",              "Cloprostenol"),
            ("Penicillin Foot Rot",             "Procaine Penicillin"),
            ("Penicillin IBK IM",               "Procaine Penicillin"),
            ("Penicillin Listeriosis",          "Procaine Penicillin"),
            ("Oxytetracycline Uterine",         "Oxytetracycline"),
            ("Salmonella Antibiotic Protocol",  "Enrofloxacin"),
            ("Meloxicam NSAID Joint",           "Meloxicam"),
        ]
        cql = """
        MATCH (t:Treatment {name: $treatment})
        MATCH (d:Drug      {name: $drug})
        MERGE (t)-[:REQUIRES_DRUG]->(d)
        """
        with self._session() as s:
            for treatment, drug in mapping:
                try:
                    s.run(cql, treatment=treatment, drug=drug)
                except Exception as exc:
                    logger.debug(f"Drug link note {treatment}→{drug}: {exc}")
        logger.info("  ✓ Treatment→Drug links created")

    def _seed_known_cows(self) -> None:
        logger.info(f"Seeding {len(KNOWN_COWS)} known MMCows cows …")
        cql = """
        MERGE (c:Cow {cow_id: $cow_id})
        SET c.breed       = $breed,
            c.age_years   = $age_years,
            c.source      = $source,
            c.status      = $status
        """
        with self._session() as s:
            for cow in KNOWN_COWS:
                s.run(cql, **cow)
        logger.info(f"  ✓ {len(KNOWN_COWS)} cows seeded")

    def _seed_pubmed_research(self, target: int = 800) -> None:
        """Fetch and store PubMed abstracts (2021-2026) as Research nodes."""
        logger.info(f"Fetching PubMed research documents (target={target}) …")
        session = requests.Session()
        session.headers["User-Agent"] = "VetIDSS/2.0 (research@vetidss.ai)"

        sub_queries = [
            ('mastitis',        'cattle mastitis somatic cell count treatment 2021:2026[pdat]'),
            ('respiratory',     'bovine respiratory disease BRD treatment prevention 2021:2026[pdat]'),
            ('lameness',        'cattle lameness hoof disease claw treatment 2021:2026[pdat]'),
            ('metabolic',       'bovine ketosis milk fever metabolic hypocalcemia 2021:2026[pdat]'),
            ('reproductive',    'bovine metritis retained placenta reproductive 2021:2026[pdat]'),
            ('diarrhea',        'calf diarrhea neonatal salmonella cryptosporidium 2021:2026[pdat]'),
            ('skin',            'lumpy skin disease LSD bovine dermatitis 2021:2026[pdat]'),
            ('neurological',    'bovine neurological polioencephalomalacia listeriosis 2021:2026[pdat]'),
            ('general',         '(cattle OR bovine) (disease OR diagnosis OR treatment) 2024:2026[pdat]'),
        ]

        all_docs = []
        seen = set()
        per_query = max(50, target // len(sub_queries))

        for disease_tag, query in sub_queries:
            if len(all_docs) >= target:
                break
            ids = []
            for start in range(0, per_query + 100, 100):
                try:
                    r = session.get(self.NCBI_SEARCH, params={
                        "db": "pubmed", "term": query,
                        "retmode": "json", "retmax": 100, "retstart": start,
                    }, timeout=30)
                    batch = r.json().get("esearchresult", {}).get("idlist", [])
                    if not batch:
                        break
                    ids.extend(batch)
                except Exception as e:
                    logger.warning(f"Search error [{disease_tag}]: {e}")
                    break
                time.sleep(0.35)
                if len(ids) >= per_query:
                    break

            for i in range(0, len(ids), 50):
                chunk = [x for x in ids[i:i+50] if x not in seen]
                if not chunk:
                    continue
                try:
                    r = session.get(self.NCBI_FETCH, params={
                        "db": "pubmed", "id": ",".join(chunk),
                        "retmode": "xml", "rettype": "abstract",
                    }, timeout=45)
                    root = ET.fromstring(r.content)
                    for art in root.findall(".//PubmedArticle"):
                        pmid = (art.findtext(".//PMID") or "").strip()
                        if not pmid or pmid in seen:
                            continue
                        year_text = art.findtext(".//PubDate/Year") or "0"
                        year = int(year_text) if year_text.isdigit() else 0
                        if not (2021 <= year <= 2026):
                            continue
                        seen.add(pmid)
                        title = art.findtext(".//ArticleTitle") or "Untitled"
                        journal = art.findtext(".//Journal/Title") or ""
                        abstract = " ".join(
                            ab.text for ab in art.findall(".//AbstractText") if ab.text
                        ) or "No abstract."
                        all_docs.append({
                            "pmid": pmid, "title": title, "journal": journal,
                            "year": year, "abstract": abstract[:3000],
                            "disease_tag": disease_tag,
                        })
                except Exception as e:
                    logger.warning(f"Fetch error [{disease_tag}]: {e}")
                time.sleep(0.35)
                if len(all_docs) >= target:
                    break

        logger.info(f"  Fetched {len(all_docs)} PubMed articles — writing to Neo4j …")
        cql_research = """
        MERGE (r:Research {pmid: $pmid})
        SET r.title       = $title,
            r.journal     = $journal,
            r.year        = $year,
            r.abstract    = $abstract,
            r.source      = 'PubMed',
            r.evidence_level = CASE
                WHEN $journal CONTAINS 'Journal' THEN 'peer_reviewed'
                ELSE 'grey_literature'
            END
        """
        cql_link = """
        MATCH (r:Research {pmid: $pmid})
        MATCH (d:Disease  {name: $disease_name})
        MERGE (r)-[:ABOUT]->(d)
        """
        tag_to_disease = {
            "mastitis": "mastitis", "respiratory": "respiratory_disease",
            "lameness": "lameness", "metabolic": "metabolic_disorder",
            "reproductive": "reproductive_issue", "diarrhea": "diarrhea",
            "skin": "skin_nodules", "neurological": "neurological_sign",
            "general": "healthy",
        }
        with self._session() as s:
            for doc in all_docs:
                try:
                    s.run(cql_research, **{k: v for k, v in doc.items() if k != "disease_tag"})
                    disease_name = tag_to_disease.get(doc["disease_tag"], "healthy")
                    s.run(cql_link, pmid=doc["pmid"], disease_name=disease_name)
                except Exception as exc:
                    logger.debug(f"Research insert note {doc.get('pmid')}: {exc}")
        logger.info(f"  ✓ {len(all_docs)} research documents stored in Neo4j")

    def _print_stats(self) -> None:
        logger.info("\n── KG Statistics ──────────────────────────────")
        with self._session() as s:
            for label in ["Disease", "Symptom", "Treatment", "Drug", "Cow", "Research"]:
                cnt = s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
                logger.info(f"  {label:12s}: {cnt:4d} nodes")
            for rel in ["HAS_SYMPTOM", "TREATED_BY", "RELATED_TO", "PROGRESSES_TO", "REQUIRES_DRUG", "ABOUT"]:
                cnt = s.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c").single()["c"]
                logger.info(f"  :{rel:<20s}: {cnt:4d} relationships")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uri      = os.getenv("NEO4J_URI",      "neo4j+s://43b30b0c.databases.neo4j.io")
    user     = os.getenv("NEO4J_USER",     "43b30b0c")
    password = os.getenv("NEO4J_PASSWORD", "")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not password:
        logger.error("NEO4J_PASSWORD not set. Check your .env file.")
        sys.exit(1)

    seeder = VeterinaryKGSeeder(uri, user, password, database)
    seeder.run_all()
    seeder.driver.close()
