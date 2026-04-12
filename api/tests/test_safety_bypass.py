import sys
import os
import asyncio
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from services.safety_engine import ClinicalSafetyEngine
from services.llm_service import GroqLLMService
from app.config import get_settings

async def test_safety_bypass_logic():
    print("Testing Clinical Safety Bypass Fix...")
    
    settings = get_settings()
    safety = ClinicalSafetyEngine()
    llm = GroqLLMService(api_key=settings.groq_api_key)
    
    # Mock predictions containing a Notifiable Disease (oral_lesion -> FMD risk)
    mock_preds = [
        {"disease": "oral_lesion", "confidence": 0.75},
        {"disease": "lymph_node_swelling", "confidence": 0.72}
    ]
    uncertainty = 0.99  # High uncertainty
    weight = 600.0
    
    # 1. Verify Safety Engine logic
    print("\n--- Phase 1: Safety Engine Logic ---")
    safety_result = safety.validate_predictions(
        disease_predictions=mock_preds,
        model_uncertainty=uncertainty,
        animal_weight_kg=weight
    )
    
    print(f"Safety Level: {safety_result['safety_level']}")
    print(f"Allow Dosing: {safety_result['allow_dosing']}")
    print(f"Notifiable Found: {safety_result['notifiable_diseases']}")
    
    assert safety_result['safety_level'] == "blocked"
    assert safety_result['allow_dosing'] is False
    assert "oral_lesion" in safety_result['notifiable_diseases']
    print("Safety Engine logic verified: allow_dosing is FALSE for blocked cases.")

    # 2. Verify LLM Service output (Mocking the prompt builder first)
    print("\n--- Phase 2: LLM Prompt Builder ---")
    prompt = llm._build_idss_prompt(
        cow_id=123,
        disease_predictions=mock_preds,
        risk_assessment={"risk_level": "high", "overall_risk_score": 0.9},
        rag_context=[],
        kg_context={},
        sensor_data={},
        animal_weight_kg=weight,
        vision_analysis={"visual_observations": ["Appears healthy"]},
        safety_status=safety_result
    )
    
    # Check if dosing tables are present
    has_dosing_tables = "2.2 × 600" in prompt or "1320 mg" in prompt
    has_safety_block_text = "CRITICAL SAFETY BLOCK" in prompt
    
    print(f"Prompt has dosing tables: {has_dosing_tables}")
    print(f"Prompt has Safety Block text: {has_safety_block_text}")
    
    assert not has_dosing_tables
    assert has_safety_block_text
    print("LLM Prompt Builder verified: dosing tables omitted and block instruction injected.")

    # 3. Final End-to-End LLM Generation (Optional, requires API)
    print("\n--- Phase 3: LLM Report Generation (Live API) ---")
    try:
        report_result = await asyncio.to_thread(
            llm.generate_clinical_report,
            cow_id=123,
            disease_predictions=mock_preds,
            risk_assessment={"risk_level": "high"},
            rag_context=[],
            animal_weight_kg=weight,
            safety_status=safety_result
        )
        report_text = report_result.get("report", "")
        
        # Verification of LLM behavior
        forbidden_keywords = ["Penicillin", "Flunixin", "Meloxicam", "mg", "MIU", "1320", "1430"]
        found_keywords = [k for k in forbidden_keywords if k.lower() in report_text.lower()]
        
        print(f"Forbidden keywords found: {found_keywords}")
        if not found_keywords:
            print("LLM Report generation verified: No medication names or dosages found.")
        else:
            print(f"FAILED: Found {found_keywords}")
            # print(f"Report Text: {report_text}")
            
    except Exception as e:
        print(f"LLM Generation skipped or failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_safety_bypass_logic())
