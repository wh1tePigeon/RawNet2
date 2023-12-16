import argparse
from pathlib import Path
import gdown


URL_LINKS = {
    "main": "https://drive.google.com/file/d/1fXtT6g4hg9y-8Az4IQEmkwHZep1gI7MJ/view?usp=sharing",
    "main_no_abs": "https://drive.google.com/file/d/1eTqIl7nsmBNWsb50gcGJrdZcm_tatde_/view?usp=sharing",
    "main_no_weight_ce": "https://drive.google.com/file/d/1dSil-oF7BG4pPRiVVfoNlLjLB3i9yrKP/view?usp=sharing",
    "main_non_zero_hz": "https://drive.google.com/file/d/1bCjIa8GXMzXdwrW_kQgvGqUpei2F2U1b/view?usp=sharing",
    "main_dif_conv": "https://drive.google.com/file/d/1fXJVCQbAB-8BUdTXlt6fdmb7M9AWLGrR/view?usp=sharing",
    "main_no_bn": "https://drive.google.com/file/d/1mf8nG9z-Zzg4p90lA-4iaVXVIxmZZRS7/view?usp=sharing",
    "main_s3" : "https://drive.google.com/file/d/1UIwTBUMzZjNmWxZfi-KvUnP6HgDyyEYw/view?usp=sharing"
}

SAVE_PATH = Path("test_model/")


def main(name):
    link = URL_LINKS[name]
    SAVE_PATH.mkdir(exist_ok=True, parents=True)
    gdown.download(url=link, output=str(SAVE_PATH / str(name + ".pth")), fuzzy=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="download model")
    parser.add_argument(
        "-n",
        "--name",
        default="main",
        type=str,
        help="name of the model (default: main)",
    )
    args = parser.parse_args()
    main(args.name)
