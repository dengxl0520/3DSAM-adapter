import os
import pickle

from torch.utils.data import DataLoader

from dataset.ct_dataset import BaseCTDataset


class BASVolumeDataset(BaseCTDataset):
    def _set_dataset_stat(self):
        self.intensity_range = (-1477.0, 786.0)
        self.target_spacing = (1, 1, 1)
        self.global_mean = -952.29052734375
        self.global_std = 87.9747543334961
        self.spatial_index = [0, 1, 2]
        self.do_dummy_2D = False
        self.target_class = 1


class ATMVolumeDataset(BaseCTDataset):
    def _set_dataset_stat(self):
        self.intensity_range = (-1263, 1903)
        self.target_spacing = (1, 1, 1)
        self.global_mean = -918.8084106445312
        self.global_std = 166.7335662841797
        self.spatial_index = [0, 1, 2]
        self.do_dummy_2D = False
        self.target_class = 1


class ParseVolumeDataset(BaseCTDataset):
    def _set_dataset_stat(self):
        self.intensity_range = (-1024, 1821)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 47.54948425292969
        self.global_std = 296.06329345703125
        self.spatial_index = [0, 1, 2]
        self.do_dummy_2D = False
        self.target_class = 1


class ImageCASVolumeDataset(BaseCTDataset):
    def _set_dataset_stat(self):
        self.intensity_range = (-987, 3069)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 148.35289001464844
        self.global_std = 179.9505615234375
        self.spatial_index = [0, 1, 2]
        self.do_dummy_2D = False
        self.target_class = 1


class Lung58VolumeDataset(BaseCTDataset):
    def _set_dataset_stat(self):
        self.intensity_range = (-1024, 6618)
        self.target_spacing = (1, 1, 1)
        self.global_mean = -713.3667602539062
        self.global_std = 311.08795166015625
        self.spatial_index = [0, 1, 2]
        self.do_dummy_2D = False
        self.target_class = 4


DATASET_DICT = {
    "bas": BASVolumeDataset,
    "atm": ATMVolumeDataset,
    "parse": ParseVolumeDataset,
    "imagecas": ImageCASVolumeDataset,
    "lung58": Lung58VolumeDataset,
}

DATASET_PATH_DICT = {
    "bas": "/data/dengxiaolong/airway/BAS/Data3/",
    "atm": "/data/dengxiaolong/airway/ATM/Data/",
    "parse": "/data/dengxiaolong/airway/PARSE/Data/",
    "imagecas": "/data/dengxiaolong/airway/ImageCAS/Data/",
    "lung58": "/data/dengxiaolong/airway/Lung58/Data/",
}


def load_data_volume(
    *,
    data,
    batch_size,
    split="train",
    deterministic=False,
    augmentation=False,
    fold=0,
    rand_crop_spatial_size=(96, 96, 96),
    convert_to_sam=False,
    do_test_crop=True,
    do_val_crop=True,
    do_nnunet_intensity_aug=False,
    num_worker=4,
):
    path_prefix = DATASET_PATH_DICT[data]

    split_file = os.listdir(path_prefix)
    split_file = [i for i in split_file if i.endswith(".pickle")][0]
    split_file = os.path.join(path_prefix, split_file)

    with open(split_file, "rb") as f:
        d = pickle.load(f)[split]

    img_files = [os.path.join(path_prefix, "images", i) for i in d]
    seg_files = [os.path.join(path_prefix, "labels", i.replace("_0000", "")) for i in d]

    dataset = DATASET_DICT[data](
        img_files,
        seg_files,
        split=split,
        augmentation=augmentation,
        rand_crop_spatial_size=rand_crop_spatial_size,
        convert_to_sam=convert_to_sam,
        do_test_crop=do_test_crop,
        do_val_crop=do_val_crop,
        do_nnunet_intensity_aug=do_nnunet_intensity_aug,
    )

    if deterministic:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_worker,
            drop_last=True,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_worker,
            drop_last=True,
        )
    return loader


if __name__ == "__main__":
    dataloader = load_data_volume(
        data="bas",
        batch_size=1,
        augmentation=True,
        split="train",
        deterministic=False,
        rand_crop_spatial_size=(128, 128, 128),
        num_worker=8,
    )
