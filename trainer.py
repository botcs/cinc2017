import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time


def accuracy(logit, target):
    y_oh = th.zeros(logit.size())
    y_oh.scatter_(1, target[:, None], 1)

    l_oh = th.eq(
        logit, logit.max(1)[0].expand(len(logit), len(logit[0]))
    ).type_as(y_oh)

    conf_mat = y_oh.t_().mm(l_oh)
    ref_sum = conf_mat.sum(0).squeeze()
    pred_sum = conf_mat.sum(1).squeeze()
    
    F1 = conf_mat.diag() * 2 / (ref_sum + pred_sum + 1e-9)
    F1 = th.cat([F1[None, :], F1[None, :].mean(dim=1)], dim=1)
    return F1


def ema(series, alpha=0.001):
    res = [series[0]]
    x = res[-1]
    for t in series[1:]:
        x = x + (t - x) * alpha
        res.append(x)

    return res


def evaluate(net, test_producer, gpu_id):
    net.eval()
    T = 0
    for i, data in enumerate(test_producer, 1):
        t = time.time()
        outputs = net(data['x'].cuda(gpu_id), data['len'].cuda(gpu_id))
        T += time.time() - t
        if i == 1:
            acc_sum = accuracy(outputs.data, data['y'].data)

        acc_sum += accuracy(outputs.data, data['y'].data)
        #print('\r%4d, sample/sec: %3.2f' % (i, len(data) / T * i), end='')
    acc = acc_sum / i
    return acc

def make_dir(save_path):
    trial = 0
    while True:
        trial += 1
        path = save_path + ('/%04d' % trial)
        if not os.path.exists(path):
            break
    os.makedirs(path)
    print('Created empty directory at:', os.path.abspath(path))
    return path

def load_latest(save_path):
    trial = 1
    while os.path.exists(save_path + ('/%04d' % (trial+1))):
        trial += 1
    path = save_path + ('/%04d' % trial)
    print('Using latest training at:', os.path.abspath(path))
    return path
            
class Trainer:
    def __init__(self, path, restore=False):
        self.restore = restore
        self.path = load_latest(path) if restore else make_dir(path)
        assert os.path.exists(self.path)
        self.losses = []
        self.train_F1 = []
        self.test_F1 = []
        self.test_highscore = 0
        self.highscore_epoch = -1

    def train(self, net, train_producer, test_producer, epochs=420, 
              gpu_id=0, useAdam=True, log2file=True):
        
        log = None
        if log2file:
            log = open(self.path + '/log', 'w')
            
        net.cuda(gpu_id)
        criterion = nn.CrossEntropyLoss()
        epoch_t_sum = 0
        if useAdam:
            learning_rate = 1e-4
        else:
            learning_rate = 1e-2
        
        if self.restore:
            net.load_state_dict(th.load(self.path+'/state_dict_highscore'))
        
        self.restore = True
        for epoch in range(len(self.test_F1)+1, epochs+1):
            if epoch % (epochs // 4) == 0:
                learning_rate /= 10.
            if useAdam:
                optimizer = optim.Adam(net.parameters(), learning_rate, weight_decay=0.0005)
            else:
                optimizer = optim.SGD(net.parameters(), learning_rate, 
                                      weight_decay=0.0005, momentum=.9)
            acc_sum = 0
            net.train()
            epoch_start = time.time()
            for i, data in enumerate(train_producer, 1):
                start_t = time.time()

                optimizer.zero_grad()
                input = data['x'].cuda(gpu_id)
                outputs = net.forward(input, data['len'].cuda(gpu_id))

                inference_t = time.time() - start_t
                loss = criterion(outputs, data['y'].cuda(gpu_id))
                loss.backward()
                optimizer.step()

                update_t = time.time() - start_t
                self.losses.append(loss.data.tolist()[0])

                self.train_F1.append(
                    accuracy(outputs.data, data['y'].data))
                acc_sum += self.train_F1[-1]
                if i % (len(train_producer) // 10) == 0:
                    stat = epoch, i, self.losses[-1], outputs.size()[0]/update_t
                    print('[%4d, %3d] loss: %5.4f\tsample/sec: %4.1f' % stat, file=log)

            
            if self.path and epoch % (epochs // 2) == 0:
                th.save(net.state_dict(), self.path+'/state_dict')
            
            th.save(self, self.path + '/' +'trainer')
            
            print('Train acc:\n', 
                  'N: %.4f  A: %.4f  O: %.4f  mean: %.4f'%
                  tuple((acc_sum/i).tolist()[0]), file=log)
            test_acc = evaluate(net, test_producer, gpu_id)
            if test_acc.tolist()[0][-1] > self.test_highscore:
                self.test_highscore = test_acc.tolist()[0][-1]
                self.highscore_epoch = epoch
                print('<<<< %.4f @ %05d epoch >>>>' % (
                    self.test_highscore, self.highscore_epoch), file=log)
                th.save(net.state_dict(), self.path+'/state_dict_highscore')
            
            print('Test acc:\n', 
                  'N: %.4f  A: %.4f  O: %.4f  mean: %.4f'%
                  tuple(test_acc.tolist()[0]), file=log)
            self.test_F1.append(test_acc)
            
            epoch_t_sum += time.time() - epoch_start
            epoch_time = epoch_t_sum / 60 / epoch
            ETL = (epochs - epoch) * epoch_time
            print('\nepoch time: %10.2f min' % epoch_time, file=log)
            print('     total: %10.2f min' % (epoch_t_sum/60), file=log)
            print(' est. left: %10.2f min' % ETL, file=log)
            print('-' * 40, file=log)
        
        print('Finished training!\n  Total time: %10.2f'%(epoch_t_sum/60), file=log)
        print('  Highscore %.4f @ %05d epoch' % (self.test_highscore, self.highscore_epoch), file=log)
        if log2file:
            print('Finished training!\n  Total time: %10.2f'%(epoch_t_sum/60))
            print('  Highscore: %.4f @ %05d epoch' % (self.test_highscore, self.highscore_epoch))
            log.close()
        return self.losses, self.train_F1, self.test_F1

    def plot(self, ema_loss=.1, ema_train_f1=.1, ema_test_f1=.8, filename=None):
        import matplotlib.pyplot as plt
        
        losses = self.losses
        F1 = self.train_F1
        test_F1 = self.test_F1
        fig1 = plt.figure()
        #plt.subplot(3, 1, 1)
        plt.plot(ema(losses, ema_loss))
        plt.title('Train Loss')
        
        #plt.subplot(3, 1, 2)
        fig2 = plt.figure()
        alpha = ema_train_f1
        plt.plot(ema(th.cat(F1)[:, 0], alpha), label='N')
        plt.plot(ema(th.cat(F1)[:, 1], alpha), label='A')
        plt.plot(ema(th.cat(F1)[:, 2], alpha), label='O')
        plt.plot(ema(th.cat(F1)[:, 3], alpha), label='Mean')
        plt.title('Train Accuracy')
        plt.legend(loc='lower right')
        
        #plt.subplot(3, 1, 3)
        fig3 = plt.figure()
        alpha = ema_test_f1
        plt.plot(ema(th.cat(test_F1)[:, 0], alpha), label='N')
        plt.plot(ema(th.cat(test_F1)[:, 1], alpha), label='A')
        plt.plot(ema(th.cat(test_F1)[:, 2], alpha), label='O')
        plt.plot(ema(th.cat(test_F1)[:, 3], alpha), label='Mean')
        plt.title('Test Accuracy')
        plt.legend(loc='lower right')
        if filename is not None:
            fig3.savefig(filename+'test.png')
        return fig1, fig2, fig3

    def __call__(self, *args, **kwargs):
        self.train(*args, **kwargs)
