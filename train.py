import argparse
import os
from shutil import copyfile
import click
import yaml
import torch

from loaders import get_loader
from learners import get_learner
from utils.tensorboard_logger import TensorboardLogger
import utils.logger
LOGGER = utils.logger.get_logger(__name__, level='DEBUG')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, default=None, help='Path for config yaml')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--from_scratch', '-f', action='store_true', default=False, help='Force training from scratch')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config_args = yaml.load(f, Loader=yaml.SafeLoader)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    # Start from scatch or resume existing model and optim
    if os.path.exists(config_args['training']['output_folder']):
        list_previous_ckpt = sorted([f for f in os.listdir(config_args['training']['output_folder'])
                                     if 'model_epoch' in f])
        if args.from_scratch or not list_previous_ckpt:
            LOGGER.info('Starting from scratch')
            if click.confirm('Removing current training directory ? ({}).'
                                     .format(config_args['training']['output_folder']), abort=True):
                os.system('rm -r ' + config_args['training']['output_folder'])
            os.mkdir(config_args['training']['output_folder'])
            start_epoch = 1
        else:
            last_ckpt = list_previous_ckpt[-1]
            checkpoint = torch.load(os.path.join(config_args['training']['output_folder'], '{}'.format(last_ckpt)))
            start_epoch = checkpoint['epoch']+1
    else:
        LOGGER.info('Starting from scratch')
        os.mkdir(config_args['training']['output_folder'])
        start_epoch = 1  
            
    # Load dataset
    LOGGER.info('Loading dataset {}'.format(config_args['data']['dataset']))
    dloader = get_loader(config_args)

    # Make loaders
    dloader.make_loaders()

    # Set learner
    LOGGER.warning('Learning type: {}'.format(config_args['training']['learner']))
    learner = get_learner(config_args, dloader.train_loader, dloader.val_loader, dloader.test_loader,
                          start_epoch, device)

    # Resume existing model or from pretrained one
    if start_epoch > 1:
        LOGGER.warning('Resuming from last checkpoint: {}'.format(last_ckpt))
        learner.model.load_state_dict(checkpoint['model_state_dict'])
        learner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    elif config_args['model']['resume']:
        LOGGER.info('Loading pretrained model from {}'.format(config_args['model']['resume']))
        if config_args['model']['resume'] == 'vgg16':
            learner.model.init_vgg16_params()
        else:
            pretrained_checkpoint = torch.load(config_args['model']['resume'])
            uncertainty_checkpoint = config_args['model'].get('uncertainty', None)
            if uncertainty_checkpoint:
                LOGGER.warning('Cloning training phase')
                learner.load_checkpoint(pretrained_checkpoint['model_state_dict'],
                                        torch.load(uncertainty_checkpoint)['model_state_dict'], strict=False)
            else:
                learner.load_checkpoint(pretrained_checkpoint['model_state_dict'], strict=False)

    # Log files
    LOGGER.info('Using model {}'.format(config_args['model']['name']))
    learner.model.print_summary(input_size=tuple([shape_i for shape_i in learner.train_loader.dataset[0][0].shape]))
    learner.tb_logger = TensorboardLogger(config_args['training']['output_folder'])
    copyfile(args.config_path, os.path.join(config_args['training']['output_folder'],'config_{}.yaml'.format(start_epoch)))
    LOGGER.info('Sending batches as {}'.format(tuple([config_args['training']['batch_size']]
                                                     + [shape_i for shape_i
                                                        in learner.train_loader.dataset[0][0].shape])))
    LOGGER.info('Saving logs in: {}'.format(config_args['training']['output_folder']))

    # Parallelize model
    nb_gpus = torch.cuda.device_count()
    if nb_gpus > 1:
        LOGGER.info('Parallelizing data to {} GPUs'.format(nb_gpus))
        learner.model = torch.nn.DataParallel(learner.model, device_ids=range(nb_gpus))

    # Set scheduler
    learner.set_scheduler()

    # Start training
    for epoch in range(start_epoch, config_args['training']['nb_epochs']+1):
        learner.train(epoch)

        
if __name__ == '__main__':
    main()
