import kagglehub
import os
import shutil
from pathlib import Path

KAGGLE_COMPETITION = 'cse-281-spring-26-scene-style-classification'
CACHE_DIR = Path.home() / '.cache' / 'kaggle_datasets'


def setup_kaggle_credentials() -> None:
   
    workspace_json = Path.cwd() / 'kaggle.json'
    home_json = Path.home() / '.kaggle' / 'kaggle.json'
    
    # If kaggle.json is in workspace, set it up
    if workspace_json.exists():
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(workspace_json, home_json)
        os.chmod(home_json, 0o600)  # Secure permissions
        print(f"[data.py] Kaggle credentials configured from workspace: {workspace_json}")
        return
    
    # Check if it's already in home directory
    if home_json.exists():
        print(f"[data.py] Using existing Kaggle credentials at: {home_json}")
        return
    
    raise FileNotFoundError(
        f"Kaggle credentials not found.\n"
        f"Place kaggle.json in one of these locations:\n"
        f"  1. {workspace_json} (workspace folder)\n"
        f"  2. {home_json} (home folder)\n"
        f"Get your API token from: kaggle.com/<username>/account"
    )


def get_kaggle_data(competition_id: str = KAGGLE_COMPETITION, cache_dir: Path = CACHE_DIR) -> Path:
  
    # Ensure credentials are set up first
    setup_kaggle_credentials()
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if data is already cached
    cached_path = cache_dir / competition_id
    if cached_path.exists() and list(cached_path.glob('**/*')):
        print(f"[data.py] Using cached dataset: {cached_path}")
        return cached_path
    
    print(f"[data.py] Downloading {competition_id} from Kaggle...")
    try:
        # Direct Kaggle download
        downloaded_path = kagglehub.competition_download(competition_id)
        print(f"[data.py] Downloaded to: {downloaded_path}")
        return Path(downloaded_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download from Kaggle: {e}\n"
            f"Ensure kaggle.json is at ~/.kaggle/kaggle.json\n"
            f"Get your API token from: kaggle.com/<username>/account"
        )


if __name__ == "__main__":
    # Direct access example
    data_path = get_kaggle_data()
    print(f"\nDataset location: {data_path}")
    
    # List available files
    if data_path.exists():
        files = list(data_path.rglob('*'))
        print(f"Files in dataset: {len([f for f in files if f.is_file()])}")
        for f in sorted(files)[:10]:  # Show first 10 items
            print(f"  - {f.relative_to(data_path)}")