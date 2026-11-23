'''
    原始nuscene的json文件过大，1.2G，将其分为多个小的
'''

import json
import ijson
from pathlib import Path
from collections import OrderedDict


def read_items(filename):
    with (INPUT_DIR / filename).open("rb") as f:
        yield from ijson.items(f, "item", use_float=True)


class SceneWriter:
    def __init__(self, table, scene_names):
        self.table = table
        self.scene_names = scene_names
        self.open_files = OrderedDict()
        self.written = set()

        for name in scene_names:
            folder = OUTPUT_DIR / name
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{table}.json").write_text(
                "[\n", encoding="utf-8"
            )

    def write(self, name, item):
        f = self.open_files.pop(name, None)

        if f is None:
            if len(self.open_files) >= MAX_OPEN_FILES:
                _, oldest = self.open_files.popitem(last=False)
                oldest.close()

            f = (OUTPUT_DIR / name / f"{self.table}.json").open(
                "a", encoding="utf-8"
            )

        self.open_files[name] = f

        if name in self.written:
            f.write(",\n")

        # json.dump(item, f, ensure_ascii=False)
        json.dump(item, f, ensure_ascii=False, indent=2)
        self.written.add(name)

    def close(self):
        for f in self.open_files.values():
            f.close()

        for name in self.scene_names:
            with (OUTPUT_DIR / name / f"{self.table}.json").open(
                "a", encoding="utf-8"
            ) as f:
                f.write("\n]\n")


def main():
    # scene_token -> scene_name
    scene_names = {
        s["token"]: s["name"]
        for s in read_items("scene.json")
    }

    # sample_token -> scene_name
    sample_to_scene = {}

    for table in ("sample", "sample_annotation", "sample_data"):
        writer = SceneWriter(table, scene_names.values())
        count = 0

        try:
            for item in read_items(f"{table}.json"):

                if table == "sample":
                    name = scene_names.get(item["scene_token"])

                    if name is not None:
                        sample_to_scene[item["token"]] = name
                else:
                    name = sample_to_scene.get(item["sample_token"])

                if name is None:
                    continue

                writer.write(name, item)
                count += 1
        finally:
            writer.close()

        print(f"{table}.json: {count:,} records saved")

    print(f"Done! {len(scene_names)} scenes created.")


if __name__ == "__main__":

    # step 1:
    INPUT_DIR = Path(r"E:\Dataset\nuScenes\data\sets\nuscenes_full\metadata\v1.0-trainval")
    OUTPUT_DIR = Path(r"E:\Dataset\nuScenes\data\sets\nuscenes_full\metadata\anno_split_new")
    MAX_OPEN_FILES = 32
    main()

    # step 2: check if the split json files are well-format

    # import json
    # from pathlib import Path
    #
    # root = Path(r"E:\Dataset\nuScenes\data\sets\nuscenes_full\metadata\anno_split_new")
    #
    # invalid_files = []
    #
    # for path in sorted(root.glob("scene-*/sample_annotation.json")):
    #     try:
    #         with path.open("r", encoding="utf-8") as f:
    #             json.load(f)
    #
    #     except (json.JSONDecodeError, UnicodeDecodeError) as e:
    #         invalid_files.append((path, str(e)))
    #         print(f"[INVALID] {path}: {e}")
    #
    # print(f"\nTotal invalid files: {len(invalid_files)}")








































