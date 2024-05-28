import subprocess

# Command to download TextBlob corpora
cmd = ['python3', '-m', 'textblob.download_corpora']

# Execute the command
try:
    subprocess.run(cmd, check=True)
    print("Corpora downloaded successfully!")
except subprocess.CalledProcessError as e:
    print(f"Error downloading corpora: {e}")
