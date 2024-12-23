import os
from dotenv import load_dotenv
from lyzr_agent_api.client import AgentAPI
from lyzr_agent_api.models.environment import EnvironmentConfig, FeatureConfig
from lyzr_agent_api.models.agents import AgentConfig
from lyzr_agent_api.models.chat import ChatRequest
import time
import requests

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
LYZR_API_KEY = os.getenv("LYZR_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize the AgentAPI client with the custom base URL
client = AgentAPI(x_api_key=LYZR_API_KEY, base_url="https://agent.api.lyzr.app")


# Step 1: Create the Environment
def create_environment():
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
            "provider": "groq",  # Specify 'groq' as the provider
            "model": "llama3-70b-8192",  # Replace with the specific model name
            "config": {
                "temperature": 0.5,  # Customize as needed
                "top_p": 0.9,       # Customize as needed
            },
            "env": {
                "GROQ_API_KEY": GROQ_API_KEY  # Ensure this is set in your .env file
            }
        },
    )
    # Send the request to create the environment
    try:
        environment = client.create_environment_endpoint(json_body=environment_config)
        print(f"Raw Environment Response: {environment}")
        environment_id = environment.get("env_id") if isinstance(environment, dict) else None
        if not environment_id:
            print("Error: Environment creation failed. Check configuration or API key.")
        else:
            print(f"Environment ID: {environment_id}")
        return environment_id
    except Exception as e:
        print(f"Error while creating environment: {e}")
        return None


# Step 2: Create the Agent
def create_agent(environment_id):
    if not environment_id:
        print("Error: Invalid environment_id. Cannot create agent.")
        return None

    agent_config = AgentConfig(
        env_id=environment_id,
        system_prompt="You are a personalized investment portfolio advisor.",
        name="InvestmentAdvisorAgent",
        agent_description="Provides personalized investment strategies based on client inputs.",
    )
    try:
        agent = client.create_agent_endpoint(json_body=agent_config)
        print(f"Raw Agent Response: {agent}")
        agent_id = agent.get("agent_id") if isinstance(agent, dict) else None
        if not agent_id:
            print("Error: Agent creation failed. Check configuration or API key.")
        else:
            print(f"Agent ID: {agent_id}")
        return agent_id
    except Exception as e:
        print(f"Error while creating agent: {e}")
        return None


# Step 3: Chat with the Agent
def chat_with_agent(agent_id, user_id: str, message: str, session_id: str):
    if not agent_id:
        print("Error: Invalid agent_id. Cannot chat with agent.")
        return None

    chat_request = ChatRequest(
        user_id=user_id,
        agent_id=agent_id,
        message=message,
        session_id=session_id,
    )
    for attempt in range(3):  # Retry up to 3 times
        try:
            print(f"Chat Request Payload: {chat_request}")
            response = client.chat_with_agent(json_body=chat_request)
            if response:
                print(f"Raw Chat Response: {response}")
                return response
            else:
                print("Empty response from the chat endpoint.")
                return None
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            time.sleep(2)  # Wait 2 seconds before retrying
    print("All attempts to chat with the agent failed.")
    return None


# Debugging Functions for Validation
def get_environment_details(env_id):
    url = f"https://agent.api.lyzr.ai/v2/environment/{env_id}"
    headers = {
        "x-api-key": LYZR_API_KEY,
        "Content-Type": "application/json",
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        print(f"Environment Details: {response.json()}")
    else:
        print(f"Failed to fetch environment details. Status: {response.status_code}, Response: {response.text}")


def get_agent_details(agent_id):
    url = f"https://agent.api.lyzr.ai/v2/agent/{agent_id}"
    headers = {
        "x-api-key": LYZR_API_KEY,
        "Content-Type": "application/json",
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        print(f"Agent Details: {response.json()}")
    else:
        print(f"Failed to fetch agent details. Status: {response.status_code}, Response: {response.text}")


# Main Execution
if __name__ == "__main__":
    # Create the environment and agent
    environment_id = create_environment()
    if not environment_id:
        print("Environment creation failed. Exiting...")
        exit(1)

    agent_id = create_agent(environment_id)
    if not agent_id:
        print("Agent creation failed. Exiting...")
        exit(1)

    # Validate environment and agent
    get_environment_details(environment_id)
    get_agent_details(agent_id)

    # Example interaction
    user_id = "user-123"
    session_id = "session-456"
    message = "What investment strategy do you recommend for a conservative investor aiming for retirement in 20 years?"

    response = chat_with_agent(agent_id, user_id, message, session_id)
    if response:
        print(f"Agent Response: {response.message if hasattr(response, 'message') else 'No message in response'}")
    else:
        print("Failed to get a response from the agent.")