import numpy as np
from rafs.model import RAFSModel

def CreateModel(gpu, args):

    model = RAFSModel(args, gpu)
    args.isMaster = gpu == 0
    model.RandomGenerator = np.random.RandomState(42)
    model.initialize_models()
    model.set_optimizers()

    if args.use_mGPU:
        model.set_multi_GPU()

    if args.load_ckpt:
        model.load_checkpoint()

    model.set_dataset()
    model.set_data_iterator()
    model.set_validation()
    model.set_loss_collector()

    if args.isMaster:
        print(f'Model {args.model_id} has successively created')
        
    return model, args