from .coco import CocoDataset
from .builder import DATASETS
@DATASETS.register_module()
class ShipRSImageNet_Level3_Combined_Categories(CocoDataset):
    CLASSES = (
        'Other Ship',
        'Warship',
        'Other Merchant',
        'Container Ship',
        'Cargo Ship',
        'Barge',
        'Fishing Vessel',
        'Oil Tanker',
        'Motorboat',
        'Dock',
    )
