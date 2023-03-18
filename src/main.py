import os
import sys
import logging
import numpy as np
import torch

from typing import Callable, Dict, List, Optional, Tuple, Union

from dataloader 
from factory import create_model, create_optimizer, create_lr_scheduler, create_loss_function, create_grad_scaler
from metrics import MetricTracker
#from utils import 


def train_network(
    model,
    device,
    trainData,
    valData,
    # train_loader,
    # val_loader,
    directory='.',
    num_epochs=200, 
    batch_size=4, 
    learning_rate=0.0001,
    patch_size=100, 
    num_masked_pixels=100*100/32.0, 
    augment=True,
    supervised=False,
    psf=None, 
    regularization = 0.0, 
    positivity_constraint = 1.0):
    #TODO group n2v, deconoise, pn2v configs separately
    '''
    Train a network using 
    
    
    '''
    args = {
        'model': model,
        'device': device,
        'trainData': trainData,
        'valData': valData,
        'directory': directory,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'patch_size': patch_size,
        'num_masked_pixels': num_masked_pixels,
        'augment': augment,
        'supervised': supervised,
        'psf': psf,
        'regularization': regularization,
        'positivity_constraint': positivity_constraint
    }

    if args.use_wandb:
        try:
            import wandb
            wandb.init(project=args.experiment_name, config=args)
            logging.info('using wandb logger')
        except ImportError:
            args.use_wandb = False
            logging.warning('wandb not installed, using default logger. try pip install wandb')
    else:
        logging.info('using default logger')

    model = create_model(args.model)
    loss_function = create_loss_function(args.loss_function)
    
    #Define optimizer #TODO getattr from config ?
    optimizer = create_optimizer(args.optimizer)#optim.Adam(model.parameters(), lr=learning_rate)

    #Define scheduler #TODO getattr from config ?
    scheduler = create_scheduler(args.scheduler)#optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

    train_single_epoch = get_training_function(args.train_algorithm)
    validate = get_validation_function(args.val_algorithm)

    running_loss = 0.0
    stepCounter=0
    dataCounter=0

    trainHist=[]
    trainHistreg = []
    valHist=[]

    val_data = []
    preds = []

    scaler = create_grad_scaler(init_scale=gradient_scale, enabled=amp)


    try:
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            logging.info(f'Starting epoch {epoch}')
            
            train_outputs = train_single_epoch(model, train_loader)
            
            losses=[]
            reg_losses = []
            
            stepCounter+=1

            # Loop over our virtual batch
            # for batch in tqdm(train_loader):
            
   

            # Perform validation step
            val_outputs = validate(model, val_loader)


            model.train(True)
            avgValLoss=np.mean(losses)
            if len(valHist)==0 or avgValLoss < np.min(np.array(valHist)):
                torch.save(model,os.path.join(directory,"best_"+'test'+".net"))
            valHist.append(avgValLoss)
            scheduler.step(avgValLoss)
            # epoch = (stepCounter / stepsPerEpoch)
            np.save(os.path.join(directory,"history_'test'.npy"), (np.array([np.arange(epoch),trainHist, trainHistreg, valHist])))

    except KeyboardInterrupt:

                
        # preds = np.stack(preds)
        # val_data = np.stack(val_data)
        # imsave(os.path.join(directory, 'preds_' + postfix +'.tif'), preds)
        # imsave(os.path.join(directory, 'valdata_' + postfix +'.tif'), val_data)

    utils.printNow('Finished Training')
    return trainHist, valHist, trainHistreg