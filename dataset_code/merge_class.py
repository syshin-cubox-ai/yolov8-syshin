import pathlib

import tqdm

root = pathlib.Path("D:/data/camera").resolve(strict=True)
class_ids = [0, 1]
before_merge_class = 1
after_merge_class = 0


def main():
    label_paths = sorted(list(root.joinpath("labels").rglob("**/*.txt")))

    for label_path in tqdm.tqdm(label_paths):
        with open(label_path, "r") as f:
            raw_label = f.readlines()

        label = []
        for label_one in raw_label:
            class_id = int(label_one.split(" ")[0])
            if class_id == before_merge_class:
                new_label = " ".join([str(after_merge_class)] + label_one.split(" ")[1:])
                label.append(new_label)
            else:
                label.append(label_one)

        with open(label_path, "w") as f:
            f.writelines(label)


if __name__ == "__main__":
    main()
