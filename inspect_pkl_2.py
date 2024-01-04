import pickle
import numpy as np
import argparse
import os 
from matplotlib import pyplot as plt

def plot_loss(save_dir, log_dir):
    log_fname = os.path.join(log_dir, 'log.pkl')
    with open(log_fname, 'rb') as f:
        data = pickle.load(f)
        np.set_printoptions(precision=4, suppress=True)
        for lst in data:
            print(lst)
            print(np.array(data[lst]))
            print(" ")

    plt.figure()
    plt.subplot(121)
    plt.title('train_MPJPE')
    plt.xlabel("Iterations")
    plt.plot(data['train_MPJPE'])

    plt.subplot(122)
    plt.title('val_MPJPE')
    plt.xlabel("Iterations")
    plt.plot(data['val_MPJPE'])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_MPJPE.png"))
    plt.close()


    plt.figure()
    plt.subplot(121)
    plt.title('train_PVE-T-SC')
    plt.xlabel("Iterations")
    plt.plot(data['train_PVE-T-SC'])

    plt.subplot(122)
    plt.title('val_PVE-T-SC')
    plt.xlabel("Iterations")
    plt.plot(data['val_PVE-T-SC'])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_PVE.png"))
    plt.close()



def plot_loss_w_baseline(save_dir, log_dir):
    log_fname = os.path.join(log_dir, 'log.pkl')
    with open(log_fname, 'rb') as f:
        data = pickle.load(f)
        np.set_printoptions(precision=4, suppress=True)
        for lst in data:
            if lst in ['val_PVE', 'val_PVE-T-SC', 'val_MPJPE']:
                print(lst)
                print(np.array(data[lst]))
                print(np.min(np.array(data[lst])))
                print(" ")
        # for lst in data:
        #     print(lst)
        #     print(np.array(data[lst][85]))
        #     print(" ")

    #log_fname = '/scratch/cq244/HierProbHuman/experiments/baseline/log.pkl'
    log_fname = '/scratches/nazgul/cq244/log.pkl'
    #log_fname = '/scratches/kyuban/cq244/hkpd/experiments/exp_001/log.pkl'
    with open(log_fname, 'rb') as f:
        baseline = pickle.load(f)

    plt.figure()
    plt.subplot(121)
    plt.title('train_PVE-SC')
    plt.xlabel("Iterations")
    
    plt.plot(baseline['train_PVE-SC'])
    plt.plot(data['train_PVE-SC'])

    plt.subplot(122)
    plt.title('val_PVE-SC')
    plt.xlabel("Iterations")
    
    plt.plot(baseline['val_PVE-SC'])
    plt.plot(data['val_PVE-SC'])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_PVESC.png"))
    plt.close()

    plt.figure()
    plt.subplot(121)
    plt.title('train_PVE-T-SC')
    plt.xlabel("Iterations")
    
    plt.plot(baseline['train_PVE-T-SC'])
    plt.plot(data['train_PVE-T-SC'])

    plt.subplot(122)
    plt.title('val_PVE-T-SC')
    plt.xlabel("Iterations")
    
    plt.plot(baseline['val_PVE-T-SC'])
    plt.plot(data['val_PVE-T-SC'])
    #plt.plot([0, 300], [0.00861, 0.00861])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_PVE-T-SC.png"))
    plt.close()


    plt.figure()
    plt.subplot(121)
    plt.title('train_MPJPE-PA')
    plt.xlabel("Iterations")
    plt.plot(baseline['train_MPJPE-PA'])
    plt.plot(data['train_MPJPE-PA'])

    plt.subplot(122)
    plt.title('val_MPJPE-PA')
    plt.xlabel("Iterations")
    
    plt.plot(baseline['val_MPJPE-PA'])
    plt.plot(data['val_MPJPE-PA'])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_MPJPEPA.png"))
    plt.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', '-L', type=str, default='/scratches/kyuban/cq244/multin_v2/experiments/multin_v5_001/',
                        help='Path to log to inspect.')
    parser.add_argument('--save_dir', '-s', type=str, default='./experiments/multin_v5_001/',
                        help='Path to log to inspect.')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    plot_loss_w_baseline(args.save_dir, args.log_dir)