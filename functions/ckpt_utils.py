import os
import hashlib
import requests
from tqdm import tqdm

# TODO not hardcode local path
# downloads the converted cifar10 checkpoint (pytorch ready) and returns path
# based on https://github.com/ermongroup/ddim - URL maintained by authors
def download_cifar10_checkpoint():
    url = "https://heibox.uni-heidelberg.de/f/869980b53bf5416c8a28/?dl=1"
    local_path = os.path.expanduser("~/.cache/diffusion_models_converted/diffusion_cifar10_model/model-790000.ckpt")
    expected_md5 = "82ed3067fd1002f5cf4c339fb80c4669"

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if not os.path.exists(local_path) or hashlib.md5(open(local_path, 'rb').read()).hexdigest() != expected_md5:
        print(f"Downloading CIFAR-10 model from {url}")
        with requests.get(url, stream=True) as r:
            total_size = int(r.headers.get("content-length", 0))
            with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                with open(local_path, "wb") as f:
                    for data in r.iter_content(chunk_size=1024):
                        f.write(data)
                        pbar.update(len(data))
    
    print(f"Checkpoint saved at: {local_path}")
    return local_path

if __name__ == "__main__":
    download_cifar10_checkpoint()
