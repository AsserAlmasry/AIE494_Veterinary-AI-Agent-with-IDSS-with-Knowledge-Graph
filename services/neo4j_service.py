"""
services/neo4j_service.py
==========================
Neo4j Aura DB service — knowledge graph queries for clinical decision support.
Wraps the official neo4j Python driver.  Connection is pooled + reusable.

v2: Added treatment protocol queries, contraindication checking, progression risk,
    zoonotic info, and auto-recording of pipeline diagnoses.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Neo4jService:
    """
    Read/write interface to the veterinary knowledge graph on Neo4j AuraDB.

    Schema
    ------
    (:Disease)   -[:HAS_SYMPTOM]->    (:Symptom)
    (:Disease)   -[:TREATED_BY]->     (:Treatment)
    (:Treatment) -[:REQUIRES_DRUG]->  (:Drug)
    (:Disease)   -[:RELATED_TO]->     (:Disease)
    (:Disease)   -[:PROGRESSES_TO]->  (:Disease)
    (:Cow)       -[:HAS_CASE]->       (:Case)
    (:Case)      -[:DIAGNOSED_AS]->   (:Disease)
    (:Research)  -[:ABOUT]->          (:Disease)
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        self.database = database
        self._driver = None
        self._connected = False
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            self._driver.verify_connectivity()
            self._connected = True
            self._ensure_schema()
            logger.info(f"Neo4jService connected to {uri}")
        except Exception as exc:
            logger.warning(
                f"Neo4j connection failed: {exc}. "
                "Knowledge graph features will be disabled."
            )

    # ── Connection management ─────────────────────────────────────────────────

    @contextmanager
    def _session(self):
        if not self._connected or self._driver is None:
            yield None
            return
        with self._driver.session(database=self.database) as session:
            yield session

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            logger.info("Neo4j connection closed")

    # ── Schema ────────────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        constraints = [
            "CREATE CONSTRAINT disease_name_unique IF NOT EXISTS FOR (d:Disease)   REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT symptom_name_unique IF NOT EXISTS FOR (s:Symptom)   REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT treatment_name_unique IF NOT EXISTS FOR (t:Treatment) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT cow_id_unique IF NOT EXISTS FOR (c:Cow)             REQUIRE c.cow_id IS UNIQUE",
            "CREATE CONSTRAINT drug_name_unique IF NOT EXISTS FOR (g:Drug)         REQUIRE g.name IS UNIQUE",
            "CREATE CONSTRAINT research_pmid_unique IF NOT EXISTS FOR (r:Research) REQUIRE r.pmid IS UNIQUE",
        ]
        with self._session() as session:
            if session is None:
                return
            for cql in constraints:
                try:
                    session.run(cql)
                except Exception as exc:
                    if "already exists" not in str(exc).lower():
                        logger.warning(f"Schema constraint: {exc}")

    # ── Primary disease context ───────────────────────────────────────────────

    def get_disease_context(self, disease_name: str) -> Dict[str, Any]:
        """Retrieve symptoms, treatments, and related cases for a disease."""
        with self._session() as session:
            if session is None:
                return {}
            try:
                result = session.run(
                    """
                    MATCH (dis:Disease {name: $disease})
                    OPTIONAL MATCH (dis)-[:HAS_SYMPTOM]->(sym:Symptom)
                    OPTIONAL MATCH (dis)-[:TREATED_BY]->(trt:Treatment)
                    OPTIONAL MATCH (cs:Case)-[:DIAGNOSED_AS]->(dis)
                    RETURN
                        dis.name           AS disease,
                        dis.severity       AS severity,
                        dis.zoonotic       AS zoonotic,
                        dis.notifiable     AS notifiable,
                        dis.description    AS description,
                        dis.prevalence_pct AS prevalence_pct,
                        dis.mortality_risk AS mortality_risk,
                        collect(DISTINCT sym.name) AS symptoms,
                        collect(DISTINCT trt.name) AS treatments,
                        count(DISTINCT cs)          AS case_count
                    """,
                    disease=disease_name,
                )
                record = result.single()
                if record:
                    return dict(record)
            except Exception as exc:
                logger.warning(f"Neo4j disease context query failed: {exc}")
        return {}

    # ── Treatment protocol ────────────────────────────────────────────────────

    def get_treatment_protocol(self, disease_name: str) -> List[Dict[str, Any]]:
        """Return first-line treatment protocols for a disease with drug info."""
        with self._session() as session:
            if session is None:
                return []
            try:
                result = session.run(
                    """
                    MATCH (dis:Disease {name: $disease})-[r:TREATED_BY]->(trt:Treatment)
                    OPTIONAL MATCH (trt)-[:REQUIRES_DRUG]->(drug:Drug)
                    RETURN
                        trt.name                   AS treatment,
                        trt.protocol               AS protocol,
                        trt.evidence_level         AS evidence_level,
                        trt.withdrawal_milk_days   AS withdrawal_milk_days,
                        trt.withdrawal_meat_days   AS withdrawal_meat_days,
                        trt.category               AS category,
                        r.first_line               AS first_line,
                        collect(DISTINCT drug.name) AS drugs,
                        collect(DISTINCT drug.withdrawal_milk_days) AS drug_milk_withdrawals
                    ORDER BY r.first_line DESC, trt.evidence_level ASC
                    LIMIT 5
                    """,
                    disease=disease_name,
                )
                return [dict(r) for r in result]
            except Exception as exc:
                logger.warning(f"Neo4j treatment protocol query failed: {exc}")
        return []

    # ── Disease progression risk ──────────────────────────────────────────────

    def get_progression_risk(self, disease_name: str) -> List[Dict[str, Any]]:
        """Return diseases this condition may progress to."""
        with self._session() as session:
            if session is None:
                return []
            try:
                result = session.run(
                    """
                    MATCH (d1:Disease {name: $disease})-[r:PROGRESSES_TO]->(d2:Disease)
                    RETURN
                        d2.name         AS progresses_to,
                        r.probability   AS probability,
                        r.time_days     AS time_days,
                        d2.severity     AS target_severity
                    ORDER BY r.probability DESC
                    """,
                    disease=disease_name,
                )
                return [dict(r) for r in result]
            except Exception as exc:
                logger.warning(f"Neo4j progression risk query failed: {exc}")
        return []

    # ── Zoonotic info ─────────────────────────────────────────────────────────

    def get_zoonotic_info(self, disease_names: List[str]) -> List[Dict[str, Any]]:
        """Return all diseases in the list that are zoonotic or notifiable."""
        with self._session() as session:
            if session is None:
                return []
            try:
                result = session.run(
                    """
                    MATCH (d:Disease)
                    WHERE d.name IN $diseases AND (d.zoonotic = true OR d.notifiable = true)
                    RETURN d.name AS disease, d.zoonotic AS zoonotic,
                           d.notifiable AS notifiable, d.description AS description
                    """,
                    diseases=disease_names,
                )
                return [dict(r) for r in result]
            except Exception as exc:
                logger.warning(f"Neo4j zoonotic query failed: {exc}")
        return []

    # ── Related diseases ──────────────────────────────────────────────────────

    def get_related_diseases(self, disease_name: str) -> List[str]:
        """Return diseases that co-occur or share symptoms."""
        with self._session() as session:
            if session is None:
                return []
            try:
                result = session.run(
                    """
                    MATCH (d1:Disease {name: $disease})-[r:RELATED_TO|HAS_SYMPTOM*1..2]-(d2:Disease)
                    WHERE d1 <> d2
                    RETURN DISTINCT d2.name AS related_disease
                    LIMIT 5
                    """,
                    disease=disease_name,
                )
                return [r["related_disease"] for r in result]
            except Exception as exc:
                logger.warning(f"Neo4j related diseases query failed: {exc}")
        return []

    # ── Cow history ───────────────────────────────────────────────────────────

    def get_cow_history(self, cow_id: int) -> List[Dict[str, Any]]:
        """Return all historical cases for a specific cow."""
        with self._session() as session:
            if session is None:
                return []
            try:
                result = session.run(
                    """
                    MATCH (c:Cow {cow_id: $cow_id})-[:HAS_CASE]->(cs:Case)-[:DIAGNOSED_AS]->(dis:Disease)
                    RETURN
                        cs.case_id    AS case_id,
                        dis.name      AS disease,
                        cs.diagnosis  AS diagnosis,
                        cs.confidence AS confidence,
                        cs.timestamp  AS timestamp
                    ORDER BY cs.timestamp DESC
                    LIMIT 10
                    """,
                    cow_id=cow_id,
                )
                return [dict(r) for r in result]
            except Exception as exc:
                logger.warning(f"Neo4j cow history query failed: {exc}")
        return []

    # ── Auto-record diagnosis ─────────────────────────────────────────────────

    def upsert_cow_case(
        self,
        cow_id: int,
        disease: str,
        confidence: float,
        case_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> bool:
        """Record a new diagnosis case for a cow. Auto-called by the pipeline."""
        import uuid
        from datetime import datetime
        case_id   = case_id or f"case_{cow_id}_{disease}_{uuid.uuid4().hex[:6]}"
        timestamp = timestamp or datetime.now().isoformat()
        with self._session() as session:
            if session is None:
                return False
            try:
                session.run(
                    """
                    MERGE (c:Cow {cow_id: $cow_id})
                    MERGE (dis:Disease {name: $disease})
                    MERGE (cs:Case {case_id: $case_id})
                    SET cs.diagnosis  = $disease,
                        cs.confidence = $confidence,
                        cs.timestamp  = $timestamp
                    MERGE (c)-[:HAS_CASE]->(cs)
                    MERGE (cs)-[:DIAGNOSED_AS]->(dis)
                    MERGE (c)-[:HAS_HISTORY]->(cs)
                    """,
                    cow_id=cow_id, disease=disease, case_id=case_id,
                    confidence=confidence, timestamp=timestamp,
                )
                return True
            except Exception as exc:
                logger.error(f"Neo4j upsert case failed: {exc}")
        return False

    # ── Research evidence ─────────────────────────────────────────────────────

    def get_research_evidence(self, disease_name: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Return recent peer-reviewed research for a disease from the KG."""
        with self._session() as session:
            if session is None:
                return []
            try:
                result = session.run(
                    """
                    MATCH (r:Research)-[:ABOUT]->(d:Disease {name: $disease})
                    WHERE r.abstract IS NOT NULL AND r.abstract <> 'No abstract.'
                    RETURN r.pmid AS pmid, r.title AS title, r.year AS year,
                           r.journal AS journal, r.abstract AS abstract
                    ORDER BY r.year DESC
                    LIMIT $limit
                    """,
                    disease=disease_name, limit=limit,
                )
                return [dict(r) for r in result]
            except Exception as exc:
                logger.warning(f"Neo4j research query failed: {exc}")
        return []

    # ── Graph statistics ──────────────────────────────────────────────────────

    def get_graph_stats(self) -> Dict[str, int]:
        """Return node and relationship counts for health check."""
        counts: Dict[str, int] = {}
        with self._session() as session:
            if session is None:
                return {"connected": 0}
            try:
                for label in ["Disease", "Symptom", "Treatment", "Drug", "Cow", "Case", "Research"]:
                    result = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
                    rec = result.single()
                    counts[label] = rec["cnt"] if rec else 0
                counts["connected"] = 1
            except Exception as exc:
                logger.warning(f"Stats query failed: {exc}")
        return counts

    # ── Legacy compat ─────────────────────────────────────────────────────────

    def build_clinical_relationships_from_documents(self, batch_size: int = 100) -> None:
        """Legacy method kept for backward compatibility — seeder script now handles this."""
        logger.info("Skipping legacy relationship builder — use scripts/seed_neo4j.py instead.")
