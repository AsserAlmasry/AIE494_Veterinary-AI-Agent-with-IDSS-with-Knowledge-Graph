"""
services/neo4j_service.py — Clinical-Decision Knowledge Graph Service
=====================================================================
Fully redesigned for actionable veterinary decision support.

New schema replaces academic citation relationships with clinical-decision edges:
  PRESENTS_WITH, TREATED_WITH, PROGRESSES_TO, CONTRAINDICATES,
  DIFFERENTIAL_OF, INDICATES_RISK_FOR, ABNORMAL_INDICATES,
  HAS_HEALTH_RECORD, DIAGNOSED_WITH
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)


class Neo4jService:
    """Thread-safe Neo4j wrapper with clinical-decision graph operations."""

    def __init__(self, uri: str, user: str, password: str, database: Optional[str] = None):
        self._driver = None
        self._database = database
        self._mock_history = {}  # Fallback history storage
        try:
            from neo4j import GraphDatabase
            # Stage 1: Try Primary (AuraDB)
            logger.info(f"Connecting to Neo4j: {uri} (User: {user})")
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            self._driver.verify_connectivity()
            logger.info(f"Neo4j Primary connected: {uri}")
        except Exception as e:
            logger.warning(f"Neo4j Primary failed: {e}. Trying Stage 2: Localhost...")
            try:
                # Stage 2: Try Localhost (Bolt) - Common if user has Desktop running
                local_uri = "bolt://localhost:7687"
                self._driver = GraphDatabase.driver(local_uri, auth=("neo4j", password))
                self._driver.verify_connectivity()
                logger.info(f"Neo4j Local connected: {local_uri}")
            except Exception:
                logger.warning("Neo4j Stage 2 failed. Using Stage 3: IN-MEMORY fallback graph.")
                self._driver = None

    def close(self):
        if self._driver: self._driver.close()

    def _run(self, query: str, **params) -> List[Dict]:
        if not self._driver: return []
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(query, **params)
                return [dict(r) for r in result]
        except Exception as e:
            logger.warning(f"Neo4j query error: {e}")
            return []

    # ── Schema Initialization ─────────────────────────────────────────────────

    def initialize_clinical_schema(self):
        """Seed the knowledge graph with bovine disease ontology."""
        diseases = self._get_disease_ontology()
        for d in diseases:
            self._run(
                "MERGE (dis:Disease {name: $name}) "
                "SET dis.category = $cat, dis.severity = $sev, dis.notifiable = $notif, dis.zoonotic = $zoo",
                name=d["name"], cat=d["category"], sev=d["severity"],
                notif=d.get("notifiable", False), zoo=d.get("zoonotic", False),
            )
            for sym in d.get("symptoms", []):
                self._run(
                    "MERGE (s:Symptom {name: $sn}) "
                    "WITH s MATCH (dis:Disease {name: $dn}) "
                    "MERGE (dis)-[:PRESENTS_WITH {body_part: $bp, visual: $vis}]->(s)",
                    sn=sym["name"], dn=d["name"], bp=sym.get("body_part", ""), vis=sym.get("visual", True),
                )
            for tx in d.get("treatments", []):
                self._run(
                    "MERGE (t:Treatment {name: $tn}) "
                    "SET t.drug = $drug, t.dosage = $dose, t.evidence = $ev "
                    "WITH t MATCH (dis:Disease {name: $dn}) "
                    "MERGE (dis)-[:TREATED_WITH {evidence_level: $ev, protocol: $proto}]->(t)",
                    tn=tx["name"], drug=tx.get("drug", ""), dose=tx.get("dosage", ""),
                    ev=tx.get("evidence_level", "C"), dn=d["name"], proto=tx.get("protocol", ""),
                )
                if tx.get("withdrawal_milk") or tx.get("withdrawal_meat"):
                    self._run(
                        "MATCH (t:Treatment {name: $tn}) "
                        "MERGE (w:WithdrawalPeriod {treatment: $tn}) "
                        "SET w.milk_days = $mk, w.meat_days = $mt "
                        "MERGE (t)-[:HAS_WITHDRAWAL]->(w)",
                        tn=tx["name"], mk=tx.get("withdrawal_milk", 0), mt=tx.get("withdrawal_meat", 0),
                    )
            for prog in d.get("progressions", []):
                self._run(
                    "MATCH (d1:Disease {name: $from}) "
                    "MERGE (d2:Disease {name: $to}) "
                    "MERGE (d1)-[:PROGRESSES_TO {probability: $prob, time_days: $td}]->(d2)",
                    **{"from": d["name"], "to": prog["to"], "prob": prog["probability"], "td": prog["time_days"]},
                )
            for diff in d.get("differentials", []):
                self._run(
                    "MATCH (d1:Disease {name: $from}) "
                    "MERGE (d2:Disease {name: $to}) "
                    "MERGE (d1)-[:DIFFERENTIAL_OF {factors: $fac}]->(d2)",
                    **{"from": d["name"], "to": diff["disease"], "fac": diff.get("factors", "")},
                )
        # Sensor → Disease mappings
        for s in self._get_sensor_disease_mappings():
            self._run(
                "MERGE (sen:Sensor {name: $sn, unit: $unit}) "
                "WITH sen MATCH (dis:Disease {name: $dn}) "
                "MERGE (sen)-[:ABNORMAL_INDICATES {direction: $dir, threshold: $thr}]->(dis)",
                sn=s["sensor"], unit=s["unit"], dn=s["disease"], dir=s["direction"], thr=s["threshold"],
            )
        logger.info("Clinical knowledge graph schema initialized")

    # ── Query Methods ─────────────────────────────────────────────────────────

    def get_disease_context(self, disease: str) -> Optional[Dict]:
        disease = disease.lower().replace(" ", "_")
        if not self._driver:
            # IN-MEMORY FALLBACK
            for d in self._get_disease_ontology():
                if d["name"] == disease:
                    return d
            return None

        rows = self._run(
            "MATCH (d:Disease {name: $n}) "
            "OPTIONAL MATCH (d)-[r1:PRESENTS_WITH]->(s:Symptom) "
            "OPTIONAL MATCH (d)-[r2:TREATED_WITH]->(t:Treatment) "
            "OPTIONAL MATCH (d)-[r3:PROGRESSES_TO]->(d2:Disease) "
            "RETURN d, collect(DISTINCT {symptom: s.name, body_part: r1.body_part}) as symptoms, "
            "collect(DISTINCT {treatment: t.name, drug: t.drug, dosage: t.dosage, evidence: r2.evidence_level, protocol: r2.protocol}) as treatments, "
            "collect(DISTINCT {disease: d2.name, probability: r3.probability, time_days: r3.time_days}) as progressions",
            n=disease,
        )
        if not rows: return None
        r = rows[0]
        d = dict(r["d"]) if r.get("d") else {}
        return {
            **d,
            "symptoms": [s for s in r.get("symptoms", []) if s.get("symptom")],
            "treatments": [t for t in r.get("treatments", []) if t.get("treatment")],
            "progressions": [p for p in r.get("progressions", []) if p.get("disease")],
        }

    def get_treatment_protocol(self, disease: str) -> List[Dict]:
        disease = disease.lower().replace(" ", "_")
        if not self._driver:
            # IN-MEMORY FALLBACK
            for d in self._get_disease_ontology():
                if d["name"] == disease:
                    return d.get("treatments", [])
            return []

        return self._run(
            "MATCH (d:Disease {name: $n})-[r:TREATED_BY]->(t:Treatment) "
            "OPTIONAL MATCH (t)-[:REQUIRES_DRUG]->(g:Drug) "
            "RETURN t.name as treatment, g.name as drug, t.protocol as protocol, "
            "r.evidence_level as evidence_level, "
            "t.withdrawal_milk_days as withdrawal_milk_days, t.withdrawal_meat_days as withdrawal_meat_days",
            n=disease,
        )

    def get_related_diseases(self, disease: str) -> List[Dict]:
        disease = disease.lower().replace(" ", "_")
        if not self._driver:
            # IN-MEMORY FALLBACK
            for d in self._get_disease_ontology():
                if d["name"] == disease:
                    return [{"disease": diff["disease"], "category": "unknown", "severity": "unknown"} for diff in d.get("differentials", [])]
            return []

        return self._run(
            "MATCH (d:Disease {name: $n})-[:RELATED_TO]-(d2:Disease) "
            "RETURN d2.name as disease, d2.category as category, d2.severity as severity "
            "LIMIT 5", n=disease,
        )
    def get_disease_research(self, disease: str) -> List[Dict]:
        """Fetch research articles related to a specific disease."""
        disease = disease.lower().replace(" ", "_")
        if not self._driver:
            return []

        return self._run(
            "MATCH (d:Disease {name: $n})<-[:ABOUT]-(r:Research) "
            "RETURN r.title as title, r.journal as journal, r.year as year, r.abstract as abstract "
            "LIMIT 3",
            n=disease,
        )

    def get_cow_history(self, cow_id: int) -> List[Dict]:
        if not self._driver:
            return self._mock_history.get(cow_id, [])

        return self._run(
            "MATCH (c:Cow {cow_id: $cid})-[:DIAGNOSED_WITH]->(diag:Diagnosis) "
            "RETURN diag.disease as disease, diag.confidence as confidence, "
            "diag.timestamp as timestamp, diag.method as method "
            "ORDER BY diag.timestamp DESC LIMIT 10", cid=cow_id,
        )

    def upsert_cow_case(self, cow_id: int, disease: str, confidence: float,
                        health_score: float = None, milk_yield: float = None, method: str = "pipeline"):
        """Record a diagnosis in the knowledge graph."""
        import datetime
        if not self._driver:
            if cow_id not in self._mock_history:
                self._mock_history[cow_id] = []
            self._mock_history[cow_id].insert(0, {
                "disease": disease, "confidence": confidence, 
                "timestamp": str(datetime.datetime.now()), "method": method
            })
            return

        self._run(
            "MERGE (c:Cow {cow_id: $cid}) "
            "CREATE (d:Diagnosis {disease: $dis, confidence: $conf, method: $meth, timestamp: datetime()}) "
            "MERGE (c)-[:DIAGNOSED_WITH]->(d)",
            cid=cow_id, dis=disease, conf=confidence, meth=method,
        )
        if health_score is not None:
            self._run(
                "MATCH (c:Cow {cow_id: $cid}) "
                "CREATE (h:HealthRecord {health_score: $hs, milk_yield: $mk, timestamp: datetime()}) "
                "MERGE (c)-[:HAS_HEALTH_RECORD]->(h)",
                cid=cow_id, hs=health_score, mk=milk_yield,
            )

    def get_sensor_risk_indicators(self, sensor_name: str) -> List[Dict]:
        if not self._driver:
            # IN-MEMORY FALLBACK
            return [s for s in self._get_sensor_disease_mappings() if s["sensor"] == sensor_name]

        return self._run(
            "MATCH (s:Sensor {name: $sn})-[r:ABNORMAL_INDICATES]->(d:Disease) "
            "RETURN d.name as disease, r.direction as direction, r.threshold as threshold",
            sn=sensor_name,
        )

    def get_graph_stats(self) -> Dict[str, Any]:
        try:
            rows = self._run(
                "MATCH (n) RETURN labels(n)[0] as label, count(n) as cnt "
                "UNION ALL MATCH ()-[r]->() RETURN type(r) as label, count(r) as cnt"
            )
            stats = {"connected": True}
            for r in rows: stats[r["label"]] = r["cnt"]
            return stats
        except: return {"connected": False}

    # ── Disease Ontology Data ─────────────────────────────────────────────────

    @staticmethod
    def _get_disease_ontology() -> List[Dict]:
        return [
            {"name": "mastitis", "category": "udder", "severity": "moderate",
             "symptoms": [{"name": "swollen_udder", "body_part": "udder"}, {"name": "abnormal_milk", "body_part": "udder"}, {"name": "fever"}],
             "treatments": [
                 {"name": "intramammary_antibiotics", "drug": "Cephapirin", "dosage": "1 tube/quarter", "evidence_level": "A", "withdrawal_milk": 3, "withdrawal_meat": 4},
                 {"name": "flunixin_nsaid", "drug": "Flunixin meglumine", "dosage": "2.2 mg/kg IV SID", "evidence_level": "A", "withdrawal_milk": 1.5, "withdrawal_meat": 4},
             ],
             "progressions": [{"to": "septicemia", "probability": 0.05, "time_days": 3}],
             "differentials": [{"disease": "udder_edema", "factors": "No bacteria in milk culture"}]},
            {"name": "lameness", "category": "musculoskeletal", "severity": "moderate",
             "symptoms": [{"name": "abnormal_gait", "body_part": "limb"}, {"name": "reluctance_to_move", "body_part": "limb"}],
             "treatments": [
                 {"name": "hoof_trimming", "evidence_level": "A", "protocol": "Functional trimming by trained operator"},
                 {"name": "meloxicam_nsaid", "drug": "Meloxicam", "dosage": "0.5 mg/kg SC SID", "evidence_level": "A", "withdrawal_milk": 3, "withdrawal_meat": 15},
             ],
             "differentials": [{"disease": "foot_rot", "factors": "Interdigital swelling and odor"}]},
            {"name": "bovine_respiratory_disease", "category": "respiratory", "severity": "high",
             "symptoms": [{"name": "cough"}, {"name": "nasal_discharge"}, {"name": "fever"}, {"name": "lethargy"}],
             "treatments": [
                 {"name": "florfenicol", "drug": "Florfenicol", "dosage": "20 mg/kg SC", "evidence_level": "A", "withdrawal_milk": 0, "withdrawal_meat": 28},
                 {"name": "tulathromycin", "drug": "Tulathromycin", "dosage": "2.5 mg/kg SC single", "evidence_level": "A", "withdrawal_milk": 0, "withdrawal_meat": 18},
             ],
             "progressions": [{"to": "pneumonia", "probability": 0.3, "time_days": 5}]},
            {"name": "ketosis", "category": "metabolic", "severity": "moderate",
             "symptoms": [{"name": "decreased_appetite"}, {"name": "weight_loss"}, {"name": "decreased_milk"}],
             "treatments": [
                 {"name": "propylene_glycol", "drug": "Propylene glycol", "dosage": "300 mL PO BID", "evidence_level": "A"},
                 {"name": "dexamethasone", "drug": "Dexamethasone", "dosage": "0.1 mg/kg IM", "evidence_level": "B", "withdrawal_milk": 0, "withdrawal_meat": 0},
             ]},
            {"name": "foot_mouth_disease", "category": "viral", "severity": "critical", "notifiable": True, "zoonotic": True,
             "symptoms": [{"name": "oral_lesion", "body_part": "mouth"}, {"name": "hoof_lesion", "body_part": "hoof"}, {"name": "drooling"}, {"name": "fever"}],
             "treatments": [{"name": "quarantine_protocol", "evidence_level": "A", "protocol": "Immediate isolation, authority notification"}],
             "differentials": [{"disease": "vesicular_stomatitis", "factors": "Serology differentiation required"}]},
            {"name": "lumpy_skin_disease", "category": "viral", "severity": "critical", "notifiable": True,
             "symptoms": [{"name": "skin_nodules", "body_part": "skin"}, {"name": "fever"}, {"name": "lymph_node_swelling"}],
             "treatments": [{"name": "vaccination_lsd", "evidence_level": "A", "protocol": "Live attenuated vaccine, ring vaccination"}]},
            {"name": "heat_stress", "category": "environmental", "severity": "variable",
             "symptoms": [{"name": "panting"}, {"name": "drooling"}, {"name": "decreased_appetite"}, {"name": "increased_water"}],
             "treatments": [
                 {"name": "cooling_protocol", "evidence_level": "A", "protocol": "Shade, fans, water misting, cold water access"},
                 {"name": "electrolyte_therapy", "drug": "Oral electrolytes", "evidence_level": "B"},
             ]},
            {"name": "milk_fever", "category": "metabolic", "severity": "high",
             "symptoms": [{"name": "recumbency"}, {"name": "muscle_tremors"}, {"name": "cold_ears"}],
             "treatments": [
                 {"name": "calcium_borogluconate", "drug": "Ca borogluconate 40%", "dosage": "400 mL slow IV", "evidence_level": "A"},
             ],
             "progressions": [{"to": "downer_cow_syndrome", "probability": 0.15, "time_days": 1}]},
        ]

    @staticmethod
    def _get_sensor_disease_mappings() -> List[Dict]:
        return [
            {"sensor": "body_temperature", "unit": "°C", "disease": "bovine_respiratory_disease", "direction": "above", "threshold": "39.5"},
            {"sensor": "body_temperature", "unit": "°C", "disease": "mastitis", "direction": "above", "threshold": "39.3"},
            {"sensor": "milk_conductivity", "unit": "mS/cm", "disease": "mastitis", "direction": "above", "threshold": "7.5"},
            {"sensor": "rumination_time", "unit": "min/day", "disease": "ketosis", "direction": "below", "threshold": "240"},
            {"sensor": "milk_yield", "unit": "kg/day", "disease": "mastitis", "direction": "below", "threshold": "15"},
            {"sensor": "THI", "unit": "index", "disease": "heat_stress", "direction": "above", "threshold": "72"},
            {"sensor": "activity_level", "unit": "steps/day", "disease": "lameness", "direction": "below", "threshold": "500"},
        ]
