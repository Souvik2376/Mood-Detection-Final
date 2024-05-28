import subprocess
import sys

# Get the full path to the Python executable
python_executable = sys.executable

# Command to download TextBlob corpora
cmd = [python_executable, '-m', 'textblob.download_corpora']

# Execute the command
try:
    subprocess.run(cmd, check=True)
    print("Corpora downloaded successfully!")
except subprocess.CalledProcessError as e:
    print(f"Error downloading corpora: {e}")

