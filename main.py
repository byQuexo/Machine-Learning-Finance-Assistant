import subprocess
import sys
import os

def run_streamlit_app(script_path, port=8501):
    """Run the Streamlit application"""
    try:
        # Run streamlit
        command = f"streamlit run {script_path} --server.port {port}"
        print(f"Starting Streamlit app: {script_path}")
        subprocess.run(command, shell=True, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Setup and run Streamlit app')
    parser.add_argument('--port', type=int, default=8501, help='Port to run Streamlit on')
    parser.add_argument('--app', type=str, default='UserInterface.py', help='Path to Streamlit app')

    args = parser.parse_args()

    run_streamlit_app(args.app, args.port)