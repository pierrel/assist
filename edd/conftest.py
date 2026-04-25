import pytest
import os
from dotenv import load_dotenv, find_dotenv


@pytest.fixture(scope="session", autouse=True)
def configure_env():
    print("Loading dev dotenv")
    if load_dotenv(find_dotenv(".dev.env")):
        print("Loaded dev dotenv")
    else:
        print("Could not find dev dotenv")
    yield
