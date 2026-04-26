from typing import List, Dict, Any

def analyze_cbt_anomalies(cbt_val: float) -> Optional[Dict[str, Any]]:
    if cbt_val > 41.0:
        return {"level": "CRITICAL", "type": "Hyperpyrexia", "msg": f"CBT {cbt_val}°C - Emergency fever"}
    elif cbt_val > 40.5:
        return {"level": "HIGH", "type": "Fever Grade 2", "msg": f"CBT {cbt_val}°C - High fever"}
    elif cbt_val > 39.5:
        return {"level": "WARN", "type": "Fever Grade 1", "msg": f"CBT {cbt_val}°C - Mild fever"}
    elif cbt_val < 38.0:
        return {"level": "WARN", "type": "Hypothermia", "msg": f"CBT {cbt_val}°C - Low body temperature"}
    return None

def run_herd_monitoring(ingestion_engine) -> List[Dict[str, Any]]:
    alerts = []
    
    # Check all 16 cows
    for i in range(1, 17):
        cow_id = f"C{i:02d}"
        
        # 1. CBT Check
        cbt_df = ingestion_engine.load_cbt(cow_id)
        if cbt_df is not None and not cbt_df.empty and 'temperature_C' in cbt_df.columns:
            latest_cbt = cbt_df['temperature_C'].iloc[-1]
            cbt_alert = analyze_cbt_anomalies(latest_cbt)
            if cbt_alert:
                cbt_alert["cow_id"] = cow_id
                alerts.append(cbt_alert)
                
        # 2. Milk Drop Analysis (>15% drop triggers Watch)
        milk_df = ingestion_engine.load_milk(cow_id)
        if milk_df is not None and len(milk_df) > 7 and 'milk_kg' in milk_df.columns:
            recent_yield = milk_df['milk_kg'].iloc[-1]
            historic_avg = milk_df['milk_kg'].tail(8).head(7).mean()
            
            if historic_avg > 0:
                drop_pct = ((historic_avg - recent_yield) / historic_avg) * 100
                if drop_pct > 30:
                    alerts.append({"cow_id": cow_id, "level": "HIGH", "type": "Production Drop", "msg": f"Milk dropped {round(drop_pct)}% (Clinical concern)"})
                elif drop_pct > 15:
                    alerts.append({"cow_id": cow_id, "level": "WARN", "type": "Production Drop", "msg": f"Milk dropped {round(drop_pct)}% (Watch)"})
                    
    # Sort alerts by severity
    severity_map = {"CRITICAL": 0, "HIGH": 1, "WARN": 2}
    alerts.sort(key=lambda x: severity_map.get(x["level"], 3))
    return alerts
