from dataset import NuScenesPTSDataset

if __name__ == "__main__":
    import torch
    NUSC_ROOT = "/ssd1/data/nuscenes/"
    NUSC_VERSION = "v1.0-mini"
    dataset = NuScenesPTSDataset(NUSC_ROOT, NUSC_VERSION, 5)

    data = dataset[0]

    for name in data:
        if isinstance(data[name], torch.Tensor):
            print(f"{name} (size): {data[name].size()}")
        else:
            print(f"{name} (value): {data[name]}")