"""
services/bovine_iq_service.py
==============================
BovineIQ Agent integration using Groq (Llama 3.3 70B).
Replaces the previous Gemini implementation.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

BOVINE_IQ_SYSTEM_PROMPT = """
You are the Veterinary AI Agent Assistant, a friendly, professional, and welcoming clinical expert with NATIVE VISION CAPABILITIES.
Your goal is to help users manage their herd health and provide veterinary guidance.

VISION CAPABILITIES:
1. **SEEING IMAGES**: You can see any image or crop the user uploads directly. Use your native vision engine (Llama 4 Scout) to analyze physical appearance, posture, udder health, and behavior.
2. **CROPPED REGIONS**: If a user provides a crop, it is a high-resolution close-up of a specific clinical area (e.g., an ear, a leg, or an udder). Focus your analysis on the details in that crop.
3. **NO TOOLS NEEDED FOR VISION**: Do not tell the user you 'need a tool' to see the image. You can see it right now!

CONVERSATIONAL GUIDELINES:
1. **BE FRIENDLY**: Use a warm, welcoming tone. Respond naturally to greetings. 🐄✨
2. **SMALL TALK IS OK**: You can engage in general conversation before any data is uploaded.
3. **GUIDE THE USER**: If no image has been analyzed yet, invite the user to upload a cow image.
4. **TOOL USAGE**: Only use diagnostic tools (`get_cow_status`, `get_cow_image`) if the user specifically asks about a cow ID in the system database.
5. **CONCISE & EXPERT**: Maintain your professional expertise.
6. **BRANDING**: DO NOT use 'BovineIQ' branding.
"""

# Dynamic Tool Factories to pass context (data pipeline, neo4j, etc.)
def build_tools(data_pipeline, neo4j_service, rag_service, inference_pipeline=None):

    @tool
    async def get_cow_status(cow_id: str) -> str:
        """Returns the actual AI model diagnostics (Core Body Temperature, milk yield, health score) by running the MMCOWS model checkpoints for a given cow (e.g., C01)."""
        if not inference_pipeline:
            return "Error: Inference Pipeline not available to run models."
        
        try:
            import re
            match = re.search(r'\d+', cow_id)
            if not match:
                return f"Error: Could not identify a Cow ID number in '{cow_id}'."
            cow_num = int(match.group(0))
            
            # Run the models directly through the pipeline! generate_report=False skips LLM call
            result = await inference_pipeline.run_full_pipeline(
                image_bytes=None, 
                cow_id_override=cow_num, 
                generate_report=False
            )
            
            # Since we provided cow_id_override, the results are in clinical_summaries
            summaries = result.get("stages", {}).get("clinical_summaries", [])
            if not summaries:
                return f"Error: Failed to process model inference for Cow #{cow_num}."
                
            s = summaries[0]
            hs = s.get("health", {}).get("health_score", "N/A")
            milk = s.get("milk", {}).get("predicted_yield_kg", "N/A")
            # heat stress model returns 'stress_level'; 'heat_stress_level' is the aliased key
            heat_dict = s.get("heat_stress", {})
            heat = heat_dict.get("heat_stress_level", heat_dict.get("stress_level", "N/A"))
            
            health_events = 0
            if neo4j_service:
                try:
                    history = neo4j_service.get_cow_history(cow_num)
                    if isinstance(history, list):
                        health_events = len(history)
                except:
                    pass
            
            return (f"AI Diagnostics for {cow_id}:\n"
                    f"- Model Predicted Health Score: {hs}\n"
                    f"- Transformer Predicted Milk Yield: {milk} kg\n"
                    f"- Heat Stress Level: {heat}\n"
                    f"- Total Health Events in Neo4j: {health_events}")
        except ValueError:
            return f"Error: Invalid cow_id {cow_id}. Must be like 'C01'."

    @tool
    def search_vet_knowledge(query: str) -> str:
        """Search the veterinary textbooks and research papers (RAG) for medical protocols, pathophysiology, treatments, and clinical signs for bovine diseases."""
        if not rag_service:
            return "Error: RAG Service not available."
        
        try:
            results = rag_service.retrieve_for_diseases([query], top_k=2)
            if not results:
                return "No relevant veterinary knowledge found."
            return "\n\n".join([f"Source: {r.get('source', 'Unknown')}\n{r.get('text', '')}" for r in results])
        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"

    @tool
    def get_cow_image(cow_id: str) -> str:
        """Retrieves an actual image of the cow from the MMCOWS visual_data dataset feed, draws a bounding box around it, and displays it on the screen."""
        import os, random
        from PIL import Image, ImageDraw, ImageFont, ImageOps
        
        try:
            class_id = str(int(cow_id.replace("C", "")))
        except ValueError:
            return f"Error: Invalid cow_id {cow_id}. Must be like 'C01'."
            
        if data_pipeline and hasattr(data_pipeline, 'visual'):
            cam1_img_dir = os.path.join(data_pipeline.visual, "images/0725/cam_1")
            cam1_lbl_dir = os.path.join(data_pipeline.visual, "labels/combined/0725/cam_1")
        else:
            cam1_img_dir = "./visual_data/images/0725/cam_1"
            cam1_lbl_dir = "./visual_data/labels/combined/0725/cam_1"
        
        if not os.path.exists(cam1_lbl_dir):
            return "Error: Visual dataset labels directory not found."
            
        # Quickly find an image containing this cow's YOLO label
        matched_files = []
        # Only scan first 500 files to avoid blocking chat for too long
        all_files = [f for f in os.listdir(cam1_lbl_dir) if f.endswith('.txt')]
        random.shuffle(all_files)
        
        for f in all_files[:500]:
            try:
                with open(os.path.join(cam1_lbl_dir, f), 'r') as fp:
                    lines = fp.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        # MMCOWS labels are 1-indexed (C01 -> 1)
                        if parts and parts[0] == class_id:
                            matched_files.append((f, parts))
                            break
            except Exception:
                pass
                
        if not matched_files:
            return f"Error: Could not find any recent frame currently showing {cow_id} in the camera view."
            
        selected_label, bbox_parts = random.choice(matched_files)
        selected_img = selected_label.replace('.txt', '.jpg')
        img_path = os.path.join(cam1_img_dir, selected_img)
        
        if not os.path.exists(img_path):
            return f"Error: Image frame {selected_img} not found."
            
        # Draw Bounding Box Using PIL
        try:
            img = Image.open(img_path)
            # Ensure image orientation matches physical pixels
            img = ImageOps.exif_transpose(img)
            draw = ImageDraw.Draw(img)
            w, h = img.size
            
            # YOLO format: class x_center y_center width height
            x_c, y_c, bw, bh = map(float, bbox_parts[1:5])
            x1 = (x_c - bw/2) * w
            y1 = (y_c - bh/2) * h
            x2 = (x_c + bw/2) * w
            y2 = (y_c + bh/2) * h
            
            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline="#34d399", width=5)
            
            # Draw Label block
            label_text = f"Cow {cow_id}"
            try:
                font = ImageFont.truetype("arial.ttf", 26)
            except (IOError, ImportError):
                font = ImageFont.load_default()
                
            rect_y1 = max(0, y1 - 35)
            draw.rectangle([x1, rect_y1, x1 + 120, y1], fill="#34d399")
            draw.text((x1 + 8, rect_y1 + 5), label_text, fill="black", font=font)
            
            # Save temp image for UI rendering
            out_dir = "./bovine_iq/temp_images"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"highlighted_{cow_id}_{selected_img}")
            
            img.save(out_path, format="JPEG", quality=90)
            
            # Return relative path for frontend to serve
            rel_path = f"/api/bovine_iq/image/{os.path.basename(out_path)}"
            return f"Successfully retrieved the visual camera feed with bounding box tracking for {cow_id}. [SHOW_IMAGE: {rel_path}]"
        except Exception as e:
            return f"Error displaying visual data boundary boxes: {str(e)}"

    @tool
    def execute_python_code(code: str) -> str:
        """
        Executes generated Python code locally and returns the stdout output and errors.
        You have FULL PERMISSION to import and use any Python library inside the script, including image processing (cv2, PIL), file I/O, data science (pandas), or ML pipelines. Do not refuse graphical or file-based requests.
        """
        import sys
        from io import StringIO
        import traceback
        
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        
        try:
            import pandas as pd
            import numpy as np
            
            exec_globals = {
                'pd': pd, 
                'np': np,
                '__builtins__': __builtins__
            }
            
            exec(code, exec_globals)
            output = redirected_output.getvalue()
            if not output:
                 output = "Code executed successfully with no stdout output."
            return output
        except Exception as e:
            error_msg = traceback.format_exc()
            return f"Execution Failed. Please debug and retry. Traceback:\n{error_msg}"
        finally:
            sys.stdout = old_stdout

    return [get_cow_status, search_vet_knowledge, get_cow_image, execute_python_code]


class BovineIQAgent:
    """
    Groq-powered (Llama 3) BovineIQ Agent using LangChain.
    """
    def __init__(self, api_key: str, data_pipeline=None, neo4j_service=None, rag_service=None, inference_pipeline=None, model_name: str = "llama-3.1-8b-instant"):
        self.api_key = api_key
        self.model_name = model_name
        
        if self.api_key:
            self.llm = ChatGroq(
                model=self.model_name,
                api_key=self.api_key,
                temperature=0.1,
                max_tokens=2048
            )
            self.tools = build_tools(data_pipeline, neo4j_service, rag_service, inference_pipeline)
            self.llm_with_tools = self.llm.bind_tools(self.tools)
            self.tools_map = {tool.name: tool for tool in self.tools}
        else:
            self.llm_with_tools = None

    async def query(self, user_input: str, history: List[Dict[str, str]], image_b64: Optional[str] = None):
        """Process user input through the LLM with native tool calling. Supports multimodal vision if image_b64 is provided."""
        if not self.llm_with_tools:
            return "Error: GROQ_API_KEY environment variable is not set correctly."
            
        try:
            # If image is present, use a vision-capable model for this turn
            current_llm = self.llm_with_tools
            if image_b64 and "vision" not in self.model_name.lower():
                # Temporarily switch to a vision-capable model for this request
                # llama-3.2 vision previews were decommissioned, switching to Llama 4 Scout (multimodal)
                vision_llm = ChatGroq(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    api_key=self.api_key,
                    temperature=0.1,
                    max_tokens=2048
                )
                current_llm = vision_llm.bind_tools(self.tools)

            # Reconstruct the conversation
            messages = [SystemMessage(content=BOVINE_IQ_SYSTEM_PROMPT)]
            for msg in history:
                role = msg.get("role", "human")
                content = msg.get("content", "")
                if role == "human" or role == "user":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))
            
            if image_b64:
                # Multi-modal content
                content = [
                    {"type": "text", "text": user_input},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ]
                messages.append(HumanMessage(content=content))
            else:
                messages.append(HumanMessage(content=user_input))
            
            return await self._run_loop(messages, [], override_llm=current_llm)
            
        except Exception as e:
            logger.error(f"Agent Error: {str(e)}", exc_info=True)
            return f"Agent Error: {str(e)}"

    async def resume(self, pending_state: dict, approved: bool):
        """Resume agent execution after code approval."""
        messages = pending_state["messages"]
        ui_tokens = pending_state.get("ui_tokens", [])
        tool_call = pending_state["tool_call"]
        
        try:
            if approved:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_result = str(await self.tools_map[tool_name].ainvoke(tool_args))
                
                match = re.search(r'\[SHOW_IMAGE:\s*(.*?)\]', tool_result)
                if match:
                    ui_tokens.append(match.group(0))
                    
                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"]
                ))
            else:
                messages.append(ToolMessage(
                    content="The user rejected the execution of this code. Acknowledge this and ask how they would like to proceed.",
                    tool_call_id=tool_call["id"]
                ))
            
            return await self._run_loop(messages, ui_tokens)
        except Exception as e:
            return f"Agent Error during resume: {str(e)}"

    async def _run_loop(self, messages: list, ui_tokens: list, override_llm: Optional[Any] = None):
        """Internal execution loop containing HITL logic."""
        llm = override_llm or self.llm_with_tools
        for _ in range(5):
            try:
                response = await llm.ainvoke(messages)
            except Exception as e:
                logger.warning(f"LLM tool invocation failed, falling back to plain text mode: {e}")
                try:
                    # Strip any previous ToolMessages that might confuse the plain LLM if tools aren't bound
                    clean_msgs = [m for m in messages if getattr(m, 'type', '') != 'tool']
                    fallback_resp = await self.llm.ainvoke(clean_msgs)
                    content = fallback_resp.content
                    if ui_tokens:
                        content += "\n\n" + "\n".join(ui_tokens)
                    return {"status": "done", "content": content}
                except Exception as e2:
                    return {"status": "done", "content": f"Agent Error: {str(e)}"}
            
            messages.append(response)
            
            if not getattr(response, 'tool_calls', None):
                content = response.content
                if ui_tokens:
                    content += "\n\n" + "\n".join(ui_tokens)
                return {"status": "done", "content": content}
                
            # Check if execution approval is needed
            pending_python_call = None
            for tool_call in response.tool_calls:
                if tool_call["name"] == "execute_python_code":
                    pending_python_call = tool_call
                    break

            if pending_python_call:
                # Execute other normal tools that were called concurrently
                for tool_call in response.tool_calls:
                    if tool_call["name"] != "execute_python_code":
                        tool_name = tool_call["name"]
                        if tool_name in self.tools_map:
                            res = str(await self.tools_map[tool_name].ainvoke(tool_call["args"]))
                            messages.append(ToolMessage(content=res, tool_call_id=tool_call["id"]))
                
                return {
                    "status": "pending",
                    "tool_call": pending_python_call,
                    "messages": messages,  # Needs state serialization for FastAPI
                    "ui_tokens": ui_tokens
                }
                
            # No approval needed, execute all normally
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                if tool_name in self.tools_map:
                    tool_result = str(await self.tools_map[tool_name].ainvoke(tool_args))
                    
                    match = re.search(r'\[SHOW_IMAGE:\s*(.*?)\]', tool_result)
                    if match:
                        ui_tokens.append(match.group(0))
                        
                    messages.append(ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"]
                    ))
                else:
                    messages.append(ToolMessage(
                        content=f"Error: Tool {tool_name} not found.",
                        tool_call_id=tool_call["id"]
                    ))
        
        return {"status": "done", "content": "Agent stopped: Reached maximum tool execution steps."}
