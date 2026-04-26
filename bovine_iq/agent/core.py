import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from .system_prompt import BOVINE_IQ_SYSTEM_PROMPT
from .tools import TOOLS
import re

class BovineIQAgent:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        if self.api_key:
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=self.api_key, max_retries=0)
            self.llm_with_tools = self.llm.bind_tools(TOOLS)
            # Create a dictionary mapping tool names to actual functions
            self.tools_map = {tool.name: tool for tool in TOOLS}
        else:
            self.llm_with_tools = None
        
    def query(self, user_input: str, history: list):
        """Process user input through the LLM with native tool calling."""
        if not self.llm_with_tools:
            return "Error: GEMINI_API_KEY environment variable is not set correctly. Please check it."
            
        try:
            # Reconstruct the conversation
            messages = [SystemMessage(content=BOVINE_IQ_SYSTEM_PROMPT)]
            for role, msg in history:
                if role == "human":
                    messages.append(HumanMessage(content=msg))
                else:
                    messages.append(AIMessage(content=msg))
            messages.append(HumanMessage(content=user_input))
            
            return self._run_loop(messages, [])
            
        except Exception as e:
            return f"Agent Error: {str(e)}"
            
    def resume(self, pending_state: dict, approved: bool):
        """Resume agent execution after code approval."""
        messages = pending_state["messages"]
        ui_tokens = pending_state.get("ui_tokens", [])
        tool_call = pending_state["tool_call"]
        
        try:
            if approved:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_result = str(self.tools_map[tool_name].invoke(tool_args))
                
                match = re.search(r'\[SHOW_IMAGE:\s*(.*?)\]', tool_result)
                if match:
                    ui_tokens.append(match.group(0))
                    
                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"]
                ))
            else:
                messages.append(ToolMessage(
                    content="The user rejected the execution of this code. Acknowledge this and ask how they would like to proceed. Make no further attempts to execute code until requested again.",
                    tool_call_id=tool_call["id"]
                ))
            
            return self._run_loop(messages, ui_tokens)
        except Exception as e:
            return f"Agent Error during resume: {str(e)}"

    def _run_loop(self, messages: list, ui_tokens: list):
        """Internal execution loop containing HITL logic."""
        for _ in range(5):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            
            if not getattr(response, 'tool_calls', None):
                # Parse the final output cleanly
                content = response.content
                if isinstance(content, list):
                    extracted = [block.get("text", "") for block in content if isinstance(block, dict) and "text" in block]
                    content = "\n\n".join([t for t in extracted if t]) or str(content)
                    
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
                            res = str(self.tools_map[tool_name].invoke(tool_call["args"]))
                            messages.append(ToolMessage(content=res, tool_call_id=tool_call["id"]))
                
                return {
                    "status": "pending",
                    "tool_call": pending_python_call,
                    "messages": messages,
                    "ui_tokens": ui_tokens
                }
                
            # No approval needed, execute all normally
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                if tool_name in self.tools_map:
                    tool_result = str(self.tools_map[tool_name].invoke(tool_args))
                    
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
