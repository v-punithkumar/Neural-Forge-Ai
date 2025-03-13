from typing import Dict, Optional

from pydantic import Field

from neural_forge_ai.trainers.common import neural_forge_aiParams


class GenericParams(neural_forge_aiParams):
    """
    GenericParams is a class that holds configuration parameters for an neural_forge_ai SpaceRunner project.

    Attributes:
        username (str): The username for your Hugging Face account.
        project_name (str): The name of the project.
        data_path (str): The file path to the dataset.
        token (str): The authentication token for accessing Hugging Face Hub.
        script_path (str): The file path to the script to be executed. Path to script.py.
        env (Optional[Dict[str, str]]): A dictionary of environment variables to be set.
        args (Optional[Dict[str, str]]): A dictionary of arguments to be passed to the script.
    """

    username: str = Field(
        None, title="Hugging Face Username", description="The username for your Hugging Face account."
    )
    project_name: str = Field("project-name", title="Project Name", description="The name of the project.")
    data_path: str = Field(None, title="Data Path", description="The file path to the dataset.")
    token: str = Field(None, title="Hub Token", description="The authentication token for accessing Hugging Face Hub.")
    script_path: str = Field(
        None, title="Script Path", description="The file path to the script to be executed. Path to script.py"
    )
    env: Optional[Dict[str, str]] = Field(
        None, title="Environment Variables", description="A dictionary of environment variables to be set."
    )
    args: Optional[Dict[str, str]] = Field(
        None, title="Arguments", description="A dictionary of arguments to be passed to the script."
    )
