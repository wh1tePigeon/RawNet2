import argparse
from pathlib import Path
import gdown


URL_LINKS = {
    "main":
    "main_no_abs":
    "main_no_weight_ce":
    "main_non_zero_hz":
    "main_dif_conv":
    "main_no_bn":
    "main_s3":
    #no valid links yet
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
