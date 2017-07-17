# IMPORTS
import data_handler
import model as M
import dilated_model as DM
import trainer as T
import torch as th
import pickle

th.multiprocessing.set_sharing_strategy('file_system')

alt_basel_DilNet = DM.DilatedFCN(1, [128, 256, 128], [[16]*3, [32]*3, [64]*3])

dataset = data_handler.DataSet(
    'data/raw/training2017/REFERENCE.csv', data_handler.load_crop,
    path='data/raw/training2017/',
    remove_noise=True, tokens='NAO')
train_set, eval_set = dataset.disjunct_split(.8)
train_producer = th.utils.data.DataLoader(
        dataset=train_set, batch_size=256, shuffle=True,
        num_workers=0, collate_fn=data_handler.batchify)
test_producer = th.utils.data.DataLoader(
        dataset=eval_set, batch_size=128, shuffle=True,
        num_workers=0, collate_fn=data_handler.batchify)
trainer = T.trainer('ckpt/cap_alt_dil')
trainer(alt_basel_DilNet, train_producer,
        test_producer, epochs=10000, gpu_id=int(0))


pickle.dump(trainer, open('cap_alt_dil.pkl', 'wb'))
pickle.dump(trainer.test_F1, open('cap_alt_dil.trF1', 'wb'))
pickle.dump(trainer.test_F1, open('cap_alt_dil.evF1', 'wb'))
