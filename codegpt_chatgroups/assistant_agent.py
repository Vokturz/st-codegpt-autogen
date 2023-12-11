from typing import Optional, List, Dict, Any
from autogen import AssistantAgent, Agent, ConversableAgent
import requests

CODEGPT_API_URL = "https://api.codegpt.co/v1"

class CodeGPTAssistantAgent(AssistantAgent):
    def __init__(self, api_key: str, agent_id: str, llm_config: Optional[Dict] = None,
                 system_message: Optional[str] = "", **kwargs):
        authorization_header = {"Authorization": f"Bearer {api_key}"}

        # Extract agent_info
        try:
            agent_info = requests.get(f"{CODEGPT_API_URL}/agent/{agent_id}",
                                      headers=authorization_header).json()
            agent_prompt = agent_info["prompt"]
            self.agent_prompt = agent_prompt
        except:
            raise ValueError(f"Invalid agent_id: {agent_id}")

        # Set CodeGPT endpoint for chat completion
        config_list = [{"base_url": CODEGPT_API_URL,
                        "api_key": api_key,
                        "model": agent_id}]
        
        if llm_config is None:
            llm_config = {}
        
        llm_config.update({"config_list": config_list,
                           "temperature": agent_info["temperature"]})
        
        # Initialize AssistantAgent
        super().__init__(
            name = agent_info["name"].replace(" ", "_"),
            llm_config = llm_config,
            system_message = system_message + " " + agent_prompt,
            **kwargs)
        
        self.register_reply(ConversableAgent, CodeGPTAssistantAgent._generate_codegpt_reply)

    def _generate_codegpt_reply(self,
                                messages: Optional[List[Dict]] = None,
                                sender: Optional[Agent] = None,
                                config: Optional[Any] = None,
                                 **kwargs):
        sys_message = self.system_message
        # Hide agent prompt since this is retrieved from CodeGPT completion API
        self.update_system_message(sys_message.replace(self.agent_prompt, "").strip()) 
        message = self.generate_reply(messages, sender, exclude=[CodeGPTAssistantAgent._generate_codegpt_reply])
        self.update_system_message(sys_message)
        return True, message
    

class CodeGPTStandaloneAssistantAgent(AssistantAgent):
    def __init__(self, codegpt_assistant: CodeGPTAssistantAgent, verbose: bool = False, **kwargs):
        assistant_prompt = codegpt_assistant.agent_prompt
        SYS_MESSAGE = f"""Given the conversation, rephrase the last user message to be related to the following prompt:
<prompt>{assistant_prompt}</prompt>"""
        
        super().__init__(
            name = codegpt_assistant.name,
            system_message=SYS_MESSAGE,
            **kwargs)

        self.register_reply(ConversableAgent, CodeGPTStandaloneAssistantAgent._generate_codegpt_reply)
        self.codegpt_assistant = codegpt_assistant
        self.verbose = verbose
    def _generate_codegpt_reply(self,
                                messages: Optional[List[Dict]] = None,
                                sender: Optional[Agent] = None,
                                config: Optional[Any] = None,
                                 **kwargs):
        
        message = self.generate_reply(messages, sender, exclude=[CodeGPTStandaloneAssistantAgent._generate_codegpt_reply])
        output_message = self.codegpt_assistant.generate_reply([{"role": "user", "content": message}],
                                                                sender,
                                                                exclude=[CodeGPTStandaloneAssistantAgent._generate_codegpt_reply])
        self.update_system_message
        if self.verbose:
            print(self.codegpt_assistant.name + " output: " +  output_message)
        return True, output_message