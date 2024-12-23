import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from lyzr_agent_api.client import AgentAPI
from lyzr_agent_api.models.environment import EnvironmentConfig, FeatureConfig
from lyzr_agent_api.models.agents import AgentConfig
from lyzr_agent_api.models.chat import ChatRequest
import time
import json

# Load environment variables from .env file
load_dotenv()

class LyzrAgentChat:
    def __init__(self):
        self.api_key = os.getenv("LYZR_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not self.api_key or not self.groq_api_key:
            raise ValueError("Missing required API keys in .env file")
            
        self.client = AgentAPI(
            x_api_key=self.api_key, 
            base_url="https://agent.api.lyzr.ai"
        )
        
    def create_environment(self) -> Optional[str]:
        environment_config = EnvironmentConfig(
            name="InvestmentAdvisorEnvironment",
            features=[
                FeatureConfig(
                    type="SHORT_TERM_MEMORY",
                    config={},
                    priority=0,
                ),
            ],
            tools=[],
            llm_config={
                "provider": "groq",
                "model": "llama3-70b-8192",
                "config": {
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "max_tokens": 1000,
                    "timeout": 30,
                },
                "env": {
                    "GROQ_API_KEY": self.groq_api_key
                }
            },
        )
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}: Creating environment...")
                environment = self.client.create_environment_endpoint(json_body=environment_config)
                
                if not isinstance(environment, dict):
                    environment = json.loads(environment)
                    
                environment_id = environment.get("env_id")
                if not environment_id:
                    raise ValueError("Environment creation response missing env_id")
                    
                print(f"Environment created successfully with ID: {environment_id}")
                return environment_id
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    print("Max retries reached. Environment creation failed.")
                    return None

    def create_agent(self, environment_id: str) -> Optional[str]:
        if not environment_id:
            raise ValueError("Invalid environment_id")
            
        agent_config = AgentConfig(
            env_id=environment_id,
            system_prompt="""You are a personalized investment portfolio advisor. 
            Provide detailed, conservative investment strategies based on client goals and risk tolerance.
            Always consider diversification, long-term growth, and risk management in your recommendations.""",
            name="InvestmentAdvisorAgent",
            agent_description="Provides personalized investment strategies based on client inputs.",
        )
        
        try:
            print("Creating agent...")
            agent = self.client.create_agent_endpoint(json_body=agent_config)
            
            if not isinstance(agent, dict):
                agent = json.loads(agent)
                
            agent_id = agent.get("agent_id")
            if not agent_id:
                raise ValueError("Agent creation response missing agent_id")
                
            print(f"Agent created successfully with ID: {agent_id}")
            return agent_id
            
        except Exception as e:
            print(f"Error creating agent: {str(e)}")
            return None

    def chat_with_agent(self, agent_id: str, user_id: str, message: str, session_id: str) -> Optional[Dict[str, Any]]:
        if not all([agent_id, user_id, message, session_id]):
            raise ValueError("Missing required parameters for chat")
            
        chat_request = ChatRequest(
            user_id=user_id,
            agent_id=agent_id,
            message=message,
            session_id=session_id,
        )
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}: Sending chat request...")
                print(f"Request details: agent_id={agent_id}, user_id={user_id}, session_id={session_id}")
                
                response = self.client.chat_with_agent(json_body=chat_request)
                
                if not response:
                    raise ValueError("Empty response received")
                    
                if isinstance(response, str):
                    response = json.loads(response)
                    
                print(f"Chat response received successfully")
                return response
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    print("Max retries reached. Chat failed.")
                    return None

def main():
    try:
        agent_chat = LyzrAgentChat()
        
        # Create environment
        environment_id = agent_chat.create_environment()
        if not environment_id:
            raise Exception("Failed to create environment")
            
        # Create agent    
        agent_id = agent_chat.create_agent(environment_id)
        if not agent_id:
            raise Exception("Failed to create agent")
            
        # Test chat
        user_id = "user-123"
        session_id = "session-456"
        message = "What investment strategy do you recommend for a conservative investor aiming for retirement in 20 years?"
        
        response = agent_chat.chat_with_agent(agent_id, user_id, message, session_id)
        if response:
            print(f"Agent Response: {response.get('message', 'No message in response')}")
        else:
            print("Chat failed")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()