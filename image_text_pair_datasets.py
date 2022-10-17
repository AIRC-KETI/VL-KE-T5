from builtins import isinstance
import os
import glob
import json
import logging
import zipfile
import functools

import datasets

logger = logging.getLogger(__name__)

_VERSION = datasets.Version("1.0.0", "")

_URL = ""

_CITATION = """\
There is no citation information
"""

_DESCRIPTION = """\
image text pair datasets
"""


IMAGE_TEXT_PAIR_DEFAULT_FEATURES=datasets.Features(
                {
                    "image": datasets.Image(),
                    "description": datasets.Value("string"),
                    "image_url": datasets.Value("string"),
                }
            )

FNAME_TRAIN = "train-filtered.json"
FNAME_VALIDATION = "validation-filtered.json"

def generator(fname, root_dir):
    for item in json.load(open(fname, "r")):
        image_path = os.path.join(root_dir, item["path"])
        description = item["description"]
        image_url = item["image_url"]
        yield {
            "image": {
                "path": image_path,
                "bytes": open(image_path, "rb").read(),
            },
            "description": description,
            "image_url": image_url,
        }

        
class ImageTextPairDataset(datasets.GeneratorBasedBuilder):
    """Image Text Pair Dataset"""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="base",
            version=_VERSION,
            description="Image Text Pair Dataset",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=IMAGE_TEXT_PAIR_DEFAULT_FEATURES,
            supervised_keys=None,  # Probably needs to be fixed.
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):

        path_kv = {
            datasets.Split.TRAIN: [os.path.join(dl_manager.manual_dir, FNAME_TRAIN), dl_manager.manual_dir],
            datasets.Split.VALIDATION: [os.path.join(dl_manager.manual_dir, FNAME_VALIDATION), dl_manager.manual_dir],
        }

        return [
                datasets.SplitGenerator(name=k, gen_kwargs={'fpath': v, 'root_dir': vv}) for k, (v, vv) in path_kv.items()
        ]

    def _generate_examples(self, fpath, root_dir):
        """Yields examples."""
        for idx, item in enumerate(generator(fpath, root_dir)):
            yield idx, item

