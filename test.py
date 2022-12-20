import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from utils.dataprocessor import dump_sample
from dataset.HDR import TestDataset
from models.Net import Net
from utils.configs import Configs
import pdb


# configurations
configs = Configs()

# test_dataset
test_dataset = TestDataset(configs=configs)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# network
model = Net()

print('cuda: %s' % torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

# adam optimizer
optimizer = optim.Adam(model.parameters(), betas=(configs.beta1, configs.beta2), lr=configs.learning_rate)

# load checkpoints
checkpoint_file = configs.checkpoint_dir + '/checkpoint.tar'
if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print("Load checkpoint %s (epoch %d)", checkpoint_file, start_epoch)
else:
    raise ModuleNotFoundError('No checkpoint files.')


def test():

    model.eval()
    for idx, data in enumerate(test_dataloader):
        img_path, in_LDRs, in_HDRs, _, _ = data
        img_path = img_path[0]
        in_LDRs = in_LDRs.to(device)
        in_HDRs = in_HDRs.to(device)
        ref_HDRs = ref_HDRs.to(device)

        with torch.no_grad():
            res = model(in_LDRs, in_HDRs)

        dump_sample(img_path, res.cpu().detach().numpy())

if __name__ == '__main__':
    test()
