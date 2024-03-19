import pytest
import torch


def test_settings():
    assert load_model == "../save/scGPT_human" # ensure the model is saved in the correct directory
    assert max_seq_len == 1536 # data preprocessing

    # model settings
    assert embsize == 512 # set embedding size
    assert d_hid == 512
    assert nlayers == 12
    assert nhead == 8
    assert n_layers_cls == 3
    assert dropout == 0.2

    # setting for optimizer
    assert batch_size == 64 
    assert eval_batch_size == 64
    assert epochs == 15

    

    assert device == torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_pert_data():
    pert_data = PertData("./data")
    pert_data.load(data_name=data_name)
    pert_data.prepare_split(split=split, seed=1)
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)
    assert pert_data.data_name == "adamson"

max_seq_len = 1536
load_model = "../save/scGPT_human" # ensure the model is saved in the correct directory
batch_size = 64 
eval_batch_size = 64
epochs = 15
embsize =512 # set embedding size
d_hid = 512
nlayers = 12
nhead = 8
n_layers_cls = 3
dropout = 0.2
