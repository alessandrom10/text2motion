from torch.utils.data import DataLoader
from dataset.dataset_diffusion.tensors import truebones_batch_collate
from dataset.dataset_diffusion.truebones.dataset.dataset import Truebones

def get_dataset_class(name):
    return Truebones

def get_dataset(num_frames, split='train', temporal_window=31, t5_name='t5-base', balanced=False, objects_subset="all", dataset_type='Truebones', similarity_threshold=0.5):
    dataset = Truebones(split=split, num_frames=num_frames, temporal_window=temporal_window, t5_name=t5_name, balanced=balanced, objects_subset=objects_subset)
    if dataset_type == 'Truebones': return dataset
    from dataset.dataset_diffusion.truebones.dataset.dataset import PairedMotionDataset
    return PairedMotionDataset(
        dataset.opt,
        dataset.motion_dataset.cond_dict,
        dataset.motion_dataset.temporal_mask_template.shape[0] - 1,
        "t5-base",
        dataset.balanced,
        similarity_threshold=similarity_threshold
    )

def get_dataset_loader(batch_size, num_frames, split='train', temporal_window=31, t5_name='t5-base', balanced=True, objects_subset="all", dataset_type='Truebones', similarity_threshold=0.5):
    dataset = get_dataset(num_frames=num_frames, split=split, temporal_window=temporal_window, t5_name=t5_name, balanced=balanced, objects_subset=objects_subset, dataset_type=dataset_type, similarity_threshold=similarity_threshold)
    collate = truebones_batch_collate
    sampler = None
    if balanced:
        from dataset.dataset_diffusion.truebones.dataset.dataset import TruebonesSampler
        sampler = TruebonesSampler(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=True if sampler is None else False,
        num_workers=8, drop_last=True, collate_fn=collate
    )
    return loader