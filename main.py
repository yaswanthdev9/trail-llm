import os
import litellm
import json
from litellm import completion
from typing import List, Dict

litellm.api_key=os.getenv("OPENAI_API_KEY")


def extract_markdown_block(response: str, block_type: str = "json") -> str:
    """Extract code block from response"""

    if not '```' in response:
        return response

    code_block = response.split('```')[1].strip()

    if code_block.startswith(block_type):
        code_block = code_block[len(block_type):].strip()

    return code_block

def generate_response(messages: List[Dict]) -> str:
    """Call LLM to get a response."""
    response = completion(
        model="openai/gpt-4o",
        messages=messages,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()

def parse_action(response: str) -> Dict:
    """Parse the LLM response into a structured action dictionary."""
    try:
        response = extract_markdown_block(response, "action")
        response_json = json.loads(response)
        if "tool_name" in response_json and "args" in response_json:
            return response_json
        else:
            return {"tool_name": "error", "args": {"message": "You must respond with a JSON tool invocation."}}
    except json.JSONDecodeError:
        return {"tool_name": "error", "args": {"message": "Invalid JSON response. You must respond with a JSON tool invocation."}}

def list_files() -> List[str]:
    """List files in the current directory."""
    return os.listdir(".")

def read_file(file_name: str) -> str:
    """Read a file's contents."""
    try:
        with open(file_name, "r") as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: {file_name} not found."
    except Exception as e:
        return f"Error: {str(e)}"

# Define system instructions (Agent Rules)
agent_rules = [{
    "role": "system",
    "content": """
You are an AI agent that can perform tasks by using available tools.

Available tools:

```json
{
    "list_files": {
        "description": "Lists all files in the current directory.",
        "parameters": {}
    },
    "read_file": {
        "description": "Reads the content of a file.",
        "parameters": {
            "file_name": {
                "type": "string",
                "description": "The name of the file to read."
            }
        }
    },
    "terminate": {
        "description": "Ends the agent loop and provides a summary of the task.",
        "parameters": {
            "message": {
                "type": "string",
                "description": "Summary message to return to the user."
            }
        }
    }
}
```

If a user asks about files, documents, or content, first list the files before reading them.

When you are done, terminate the conversation by using the "terminate" tool and I will provide the results to the user.

Important!!! Every response MUST have an action.
You must ALWAYS respond in this format:

<Stop and think step by step. Parameters map to args. Insert a rich description of your step by step thoughts here.>

```action
{
    "tool_name": "insert tool_name",
    "args": {...fill in any required arguments here...}
}
```"""
}]

# Initialize agent parameters
iterations = 0
max_iterations = 10

user_task = input("What would you like me to do? ")

memory = [{"role": "user", "content": user_task}]

# The Agent Loop
while iterations < max_iterations:
    # 1. Construct prompt: Combine agent rules with memory
    prompt = agent_rules + memory

    # 2. Generate response from LLM
    print("Agent thinking...")

    print("++++++++++++++++++++++++++++++++++++++++")

    print("Agent prompt:" , prompt)

    print("*****************************************")

    response = generate_response(prompt)

    print(f"Agent response: {response}")

    # 3. Parse response to determine action
    action = parse_action(response)
    result = "Action executed"

    if action["tool_name"] == "list_files":
        result = {"result": list_files()}
    elif action["tool_name"] == "read_file":
        result = {"result": read_file(action["args"]["file_name"])}
    elif action["tool_name"] == "error":
        result = {"error": action["args"]["message"]}
    elif action["tool_name"] == "terminate":
        print(action["args"]["message"])
        break
    else:
        result = {"error": "Unknown action: " + action["tool_name"]}

    print(f"Action result: {result}")

    # 5. Update memory with response and results
    memory.extend([
        {"role": "assistant", "content": response},
        {"role": "user", "content": json.dumps(result)}
    ])

    # 6. Check termination condition
    if action["tool_name"] == "terminate":
        break

    iterations += 1
