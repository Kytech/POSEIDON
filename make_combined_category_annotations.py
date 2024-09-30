import os
import sys
import json
from pathlib import Path

from shiprsimagenet import ShipRSImageNet

new_class_names = {
    1: 'Other Ship',
    2: 'Warship',
    3: 'Other Merchant',
    4: 'Container Ship',
    5: 'Cargo Ship',
    6: 'Barge',
    7: 'Fishing Vessel',
    8: 'Oil Tanker',
    9: 'Motorboat',
    10: 'Dock',
}

class_id_mappings = {
    1: 1,
    **{x: 2 for x in range(2, 37)},
    37: 3,
    38: 4,
    39: 3,
    40: 5,
    41: 6,
    **{x: 3 for x in range(42,46)},
    46: 7,
    47: 8,
    48: 3,
    49: 9,
    50: 10,
}

new_coco_categories = [ { 'id': id, 'name': name, 'supercategory': name } for id, name in new_class_names.items() ]

def main():
    if len(sys.argv) < 2:
        print("Usage: python make_combined_category_annotations.py <path to ShipRSImageNet dataset>")
        sys.exit(1)
    
    if not os.path.exists(sys.argv[1]):
        print("Path does not exist")
        sys.exit(1)
    
    if not os.path.isdir(sys.argv[1]):
        print("Path to dataset is not a directory")
        sys.exit(1)
    
    print("Loading dataset...")

    dataset = ShipRSImageNet(sys.argv[1])
    train_annotations_path = Path(dataset.coco_root_dir) / dataset.get_coco_annotation_file_name('train')
    val_annotations_path = Path(dataset.coco_root_dir) / dataset.get_coco_annotation_file_name('val')
    
    print("Making combined category annotations...")
    
    make_combined_category_annotations(train_annotations_path)
    make_combined_category_annotations(val_annotations_path)
    
    print(f"Saved combined annotations in directory {dataset.coco_root_dir}")


def make_combined_category_annotations(annotation_file: Path):
    with annotation_file.open('r') as f:
        annotations = json.load(f)
    
    annotations['categories'] = new_coco_categories
    
    for annotation in annotations['annotations']:
        annotation['category_id'] = class_id_mappings[annotation['category_id']]
    
    with annotation_file.parent.joinpath(f"{annotation_file.stem}_combined_categories.json").open('w') as f:
        json.dump(annotations, f)


if __name__ == "__main__":
    main()
