from app import ODServiceOverNetworkClient
from maverick.object_detection import cli_tools

if __name__ == '__main__':
    cli_tools.launch(ODServiceOverNetworkClient('http://localhost:5000'))
