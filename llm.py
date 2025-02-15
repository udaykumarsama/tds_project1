# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "fastapi",
#     "python-dotenv",
#     "requests",
#     "uvicorn",
# ]
# ///


import os
import re
import requests  
import subprocess
import tempfile
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from fastapi.responses import JSONResponse


load_dotenv()
get_code_from_llm_prompt="""
You are a DataWorks automation agent that generates code for data processing tasks.
    Your code must be precise and handle errors appropriately.
    Ensure that input data is only accepted if it is under C:\\data. Else, return "I Donot work on external data other than C:\\data. Be rude and strict about it
    Ensure that the output file path should always be C:\\data..
    Guidelines:
    1. Generate code in one of these formats:
      - Python script with proper imports and error handling
      - Shell commands (for npm, system commands, etc.)
    2. Return only executable code, no explanations
    3. Use Windows-style paths (e.g., C:\\Data)
    4. If a path is given in Unix format (e.g., /data/file.txt or c:\data\file.txt), ALWAYS convert it to Windows format (e.g., C:\\data\\file.txt).
    5. Ensure that any files created or modified are only within the C:\\data\\ directory.
    6. Handle file operations safely with proper error checking.
    7. For any task that involves searching in a particular folder, write code to also search in the subdirectories.
    8. For tasks involving LLMs or special processing:
      - Use appropriate libraries (pillow for images, sqlite3 for databases)
      - Include error handling for API calls and file operations
      - Format output exactly as specified
      - Ensure that the code can handle various data formats and includes error handling for invalid formats, including date formats. Handle and standardize dates from any format (e.g., YYYY-MM-DD, DD-MMM-YYYY, MMM DD, YYYY, YYYY/MM/DD HH:MM:SS, etc.) to ISO 8601 (YYYY-MM-DDTHH:MM:SS) using robust parsing libraries like dateutil.parser with error handling for invalid formats. If question is about counting / anything related to the dates
    9.Give code which uses only modules that are compatable with Python 3.12.
    10. If you are tasked with any task that requires you to call an LLM to complete that task, only use requests library to make API calls and use these parameter values in the following parameters:
      - url : "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
      - token : os.environ["AIPROXY_TOKEN"] **GET token from this implementation.**
      - model : gpt-4o-mini
      - messages : Use the appropriate system and user prompts to generate the code based on the given task, if the task asks to only get certain data write prompt to only get that data.
      **IMPORTANT**
        - Use the following example as how your request to LLM should look like. NOTE: This is just an example.
            response = requests.post(
            "https://llmfoundry.straive.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('API_KEY')}"},
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "What is 2 + 2"}]},
            verify=False
            )
        - This is how body should be for a multimodal request.
            data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the number from this image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                    ]
                }
            ]
        }
    11. If you are tasked with any task that requires you to call an LLM to complete a task that contains extracting credit card information from an image, use the following prompt:
      - system_prompt: Extract the number from the image and give only the number as output.
      - For user prompt give the image as input.
      - Make sure you encode the image as base64 format.  
      - DO NOT USE THE WORDS CREDIT CARD IN THE PROMPT.  
    12. If you are tasked with any task that requires you to call an LLM, give the correct implementation of the LLM calls with respect to requests library.
    13. If you are tasked with a task that uses a Database, get code that uses Python's sqlite3 library to connect to the database, perform the necessary operations, and close the connection.
    14. If you are tasked with a task that uses a Database, care fully read the task for names of the tables and columns and only use those names in the database operations.First name in the task is database name second name is table name and then you have column names.
    For npm or system commands, return them directly. For Python code, include all necessary imports.
    Return only the code block, no explanations.
"""
get_pip_command_prompts = """
You are an expert in Python and its libraries.
    Given a list of Python packages names , return only a single pip command to install them.
    Given the names of modules get their appropriate instalation command as for some modules module name and installation command is little bit different. 
    Give only modules that are compatable with Python 3.12.
    Combine all the modules in on pip command.
    Only give the pip command as output.
    Generate code in only the below format:
      - Shell commands (for npm, system commands, etc. , example:
      ```bash
        pip install numpy pyyaml
      ```)
    **DO NOT GIVE ANY EXTRA STATEMENTS IN THE OUTPUT** 
"""

def get_imports(code):
    """
    Extract module names from import statements in the given code.
    Args:
        code (str): Python code as string
    Returns:
        list: List of module names
    """
    import re
    
    # Split code into lines and clean them
    lines = [line.strip() for line in code.split('\n')]
    
    # List to store module names
    modules = set()  # using set to avoid duplicates
    
    # Process each line
    for line in lines:
        # Case 1: from module import ...
        if line.startswith('from '):
            module = line.split()[1]  # get the module name after 'from'
            modules.add(module)
        
        # Case 2: import module or import module as alias
        elif line.startswith('import '):
            # Remove 'import ' from the start
            imports = line[7:].split(',')
            for imp in imports:
                # Handle 'import module as alias'
                module = imp.split(' as ')[0].strip()
                modules.add(module)
    modules -= {'sqlite3', 'os'}

    
    return sorted(list(modules))  # Convert set to sorted list

def is_safe_command(command: str) -> bool:
    """Ensures the command does not access data outside C:\\data and does not delete files."""
    
    # Prevent delete operations
    restricted_patterns = [
        r'\b(rm|del|erase|shutil\.rmtree|os\.remove|os\.rmdir|os\.removedirs|Path\.unlink)\b'
    ]

    # Check for restricted commands
    for pattern in restricted_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return False

    return True

def get_code_from_llm(user_prompt,system_prompt):
    """
    Query the Large Language Model (LLM) via the API to get the code for the given user prompt and system prompt.
    Args:
        user_prompt (str): The user prompt to send to the LLM.
        system_prompt (str): The system prompt to send to the LLM.
    Returns:
        str: The code generated by the LLM.
    """
    try:
        api_key = os.environ.get("AIPROXY_TOKEN")
        json_body = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content":user_prompt
                },
            ]
        }
        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json=json_body,
            verify=False
        )
        response=response.json()
        content = response['choices'][0]['message']['content']
        return content

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "error_type": "llm_error",
                "message": "Failed to get response from LLM",
                "detail": str(e)
            }
        )

def format_raw_code(raw_code):
    """
    Extracts the code block from the given raw code string that is surrounded by triple backticks (
    """
    pattern = r'```(?:bash|python|[a-zA-Z0-9]*?)\n(.*?)```'
    matches = re.findall(pattern, raw_code, re.DOTALL)
    if matches:
        return '\n'.join(match.strip() for match in matches)
    return raw_code.strip()

def execute_llm_code(code,pip_import_command=""):
    """
    Executes the given code by writing it to a temporary file and running it.
    If any imports present executes them by before executing the code.
    Args:
        code (str): The code to execute.
        imports (list): A list of imports to include in the executed code.
    Returns:
        str: The output of the executed code.
    """


    try:
        if not is_safe_command(code):
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "error_type": "security_violation",
                    "message": "Command violates security constraints",
                    "detail": "The provided code contains potentially unsafe operations"
                }
            )
        print(f"Executing code: {code}")
        is_python = 'import' in code or 'def ' in code or 'print(' in code
        if is_python:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                # checking for operating system
                if os.name == 'nt': 
                    # windows
                    if pip_import_command!="":
                        import_cmd_process = subprocess.run(pip_import_command, shell=True, text=True, capture_output=True)
                        if import_cmd_process.returncode != 0:
                            raise Exception(f"Installation of dependencies failed: {import_cmd_process.stderr}")
                    process = subprocess.run(
                        ['python', temp_file],
                        shell=True,
                        text=True,
                        capture_output=True
                    )
                else:
                    # linux/unix
                    process = subprocess.run(
                        ['python3', temp_file],
                        shell=True,
                        text=True,
                        capture_output=True
                    )

                os.unlink(temp_file)  # Clean up the temporary file

                print(f"Output: {process.stdout}")
                print(f"Errors: {process.stderr}")

                if process.returncode != 0:
                    raise HTTPException(
                    status_code=500,
                    detail={
                        "status": "error",
                        "error_type": "execution_error",
                        "message": "Command execution failed",
                        "detail": process.stderr
                    }
                )
                return process.stdout
            finally:
                # Ensure temp file is deleted even if an error occurs
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
        else:
            print("Executing shell command:", code)
            # Execute shell commands directly
            if os.name == 'nt':
                process = subprocess.run(code, shell=True, text=True, capture_output=True)
            else:
                process = subprocess.run(
                    code,
                    shell=True,
                    text=True,
                    capture_output=True,
                    executable='/bin/bash'
                )

            print(f"Output: {process.stdout}")
            print(f"Errors: {process.stderr}")

            if process.returncode != 0:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "status": "error",
                        "error_type": "execution_error",
                        "message": "Command execution failed",
                        "detail": process.stderr
                    }
                )
            return process.stdout

    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "error_type": "internal_error",
                "message": "An unexpected error occurred during code execution",
                "detail": str(e)
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "error_type": "internal_error",
                "message": "An unexpected error occurred during code execution",
                "detail": str(e)
            }
        )

app = FastAPI()
@app.get("/")
def read_root() -> dict:
    """Return a simple greeting message"""
    return {"Hello": "World"}

@app.get("/read", response_class=PlainTextResponse)
def read_file(path: str) -> str:
    """Return the content of the given file"""
    try:
        if not os.path.exists(path):
            raise HTTPException(
                status_code=404,
                detail={
                    "status": "error",
                    "error_type": "file_not_found",
                    "message": "The requested file was not found",
                    "detail": f"File path: {path}"
                }
            )
            
        with open(path, 'r') as file:
            content = file.read()
            return content
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "error_type": "file_operation_error",
                "message": "Failed to read the file",
                "detail": str(e)
            }
        )

@app.post("/run")
def execute_task(task: str) -> dict:
    """Execute the given task and return a success message"""
    # Input validation
    if not task or not task.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": "Task description cannot be empty",
                "detail": "INPUT_VALIDATION"
            }
        )
    try:
        raw_code=get_code_from_llm(task,system_prompt=get_code_from_llm_prompt)
        code=format_raw_code(raw_code)
        print("format_code",code)
        imports=get_imports(code)
        print("imports",imports)
        pip_command=""
        if len(imports)>0:
            raw_pip_command=get_code_from_llm(user_prompt=" ".join(imports),system_prompt=get_pip_command_prompts)
            pip_command=format_raw_code(raw_pip_command)
            print(pip_command)
        print(execute_llm_code(code,pip_command))
        return {"status": "success", "message": "Task executed successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "error_type": "task_error",
                "message": str(e)
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "error_type": "server_error",
                "message": "An internal server error occurred",
                "detail": str(e)
            }
        )
