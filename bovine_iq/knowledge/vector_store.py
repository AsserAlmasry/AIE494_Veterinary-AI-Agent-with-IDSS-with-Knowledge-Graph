import chromadb
from chromadb.config import Settings

class VetVectorStore:
    def __init__(self, db_dir: str = "./chroma_db"):
        self.db_dir = db_dir
        self.client = chromadb.PersistentClient(path=self.db_dir)
        # Create or get the veterinary collection
        self.collection = self.client.get_or_create_collection(name="vet_knowledge")
        self._seed_database_if_empty()
        
    def _seed_database_if_empty(self):
        if self.collection.count() == 0:
            print("Populating Veterinary Knowledge Base (RAG)...")
            docs = [
                {
                    "id": "dis_ketosis",
                    "text": "Ketosis (Metabolic): Etiology: Negative energy balance in early lactation. Pathophysiology: Mobilization of NEFA from fat stores. Liver converts NEFA to ketones (BHB, Acetoacetate). Clinical signs: Drop in milk yield, lethargy, acetone breath. Subclinical: BHB > 1.2 mmol/L. Sensor: Activity drop by 20%, CBT normal. Treatment: Propylene glycol 300g PO for 3-5 days. If severe: 500mL 50% Dextrose IV.",
                    "meta": {"category": "Metabolic", "disease": "Ketosis"}
                },
                {
                    "id": "dis_mastitis",
                    "text": "Mastitis (Infectious): Etiology: E. coli, Staph aureus, Strep uberis. Pathophysiology: Pathogens enter teat canal, endotoxin release (LPS) triggers cytokine cascade. Clinical signs: Swollen quarter, abnormal milk, fever (CBT > 40.0C). Sensor: Milk yield drop >30%, CBT >40.5C (fever grade 2), Lying time drops. Treatment: Intramammary antibiotics (e.g., Ceftiofur). Systemic NSAIDs (Flunixin meglumine 2.2mg/kg IV) to reduce fever.",
                    "meta": {"category": "Infectious", "disease": "Mastitis"}
                },
                {
                    "id": "dis_hypocalcemia",
                    "text": "Milk Fever / Hypocalcemia (Metabolic): Etiology: High calcium demand at calving. Pathophysiology: Failure of PTH and Vitamin D to mobilize Ca fast enough. Clinical signs: Muscle tremors, weakness, S-curve neck, pathological recumbency. Sensor: Lying time >16h (emergency recumbency), CBT <38.0C (mild hypothermia). Treatment: 500mL 23% Calcium Gluconate IV slowly.",
                    "meta": {"category": "Metabolic", "disease": "Milk Fever"}
                },
                {
                    "id": "dis_lameness",
                    "text": "Digital Dermatitis / Lameness (Locomotion): Etiology: Treponema species. Clinical signs: Strawberry-like lesions on heel bulbs, altered gait. Sensor: Lying time <8h (insufficient rest due to pain at feedbunk) OR lying time >14h depending on lesion. Activity_mag drop 40%. Treatment: Topical oxytetracycline spray, copper sulfate footbaths.",
                    "meta": {"category": "Lameness", "disease": "Digital Dermatitis"}
                }
            ]
            self.collection.add(
                ids=[d["id"] for d in docs],
                documents=[d["text"] for d in docs],
                metadatas=[d["meta"] for d in docs]
            )

    def search(self, query: str, k: int = 2) -> str:
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        if not results["documents"][0]:
            return "No relevant veterinary literature found."
            
        context = ""
        for idx, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][idx]
            context += f"Source ({meta['category']} - {meta['disease']}):\\n{doc}\\n\\n"
        return context
