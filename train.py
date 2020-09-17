#!/usr/bin/env python
from utils import load_dataset, get_models, load_model
import os
import logging
import numpy as np
from pprint import pprint
import torch
from random import seed
from config import args

def run(args):
    pprint(args)
    logging.basicConfig(level=logging.INFO)

    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    seed(args['seed'])

    dataset, ontology, vocab, Eword, init_state = load_dataset()

    model = load_model(args['model'], args, ontology, vocab)
    model.save_config()
    model.load_emb(Eword)
    model.initialize_state(init_state)

    model = model.to(model.device)
    if args['resume']:
        model.load(args['resume']) 
    else:
        model.init_weights()            
    if not args['test']:
        logging.info('Starting train')
        model.run_train(dataset['train'], dataset['dev'], args)
    model.load_best_save(directory=args['dout'])        
    model = model.to(model.device)
    logging.info('Running dev evaluation')
    dev_out = model.run_eval(dataset['dev'], args)
    pprint(dev_out)
    test_out = model.run_eval(dataset['test'], args)
    pprint(test_out)
    with open('seedout.txt', 'ab') as f:
        f.write('\t'.join(['seed:', str(args['seed']), 'dev goal:', str(dev_out['joint_goal']), 'test goal:', str(test_out['joint_goal'])])+'\n')
    logging.info('Making predictions for {} dialogues and {} turns'.format(len(dataset['test']), len(list(dataset['test'].iter_turns()))))
    preds = model.run_pred(dataset['test'], args)
    eval_pred = dataset['test'].evaluate_preds(preds)
    with open(os.path.join(args['dout'], '%s_log.txt'%'test'), 'wb') as f:
        f.write(str(eval_pred))
    pprint(eval_pred)    

if __name__ == '__main__':
    run(args)
