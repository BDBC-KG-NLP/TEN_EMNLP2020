import os
import json
import logging
from config import args
from pprint import pprint
from utils import load_dataset, load_model

eva_set = 'test'

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    with open(os.path.join(args['dout'], 'config.json')) as f:
        args_save = json.load(f)
        args_save['device'] = args['device']
    pprint(args_save)

    dataset, ontology, vocab, Eword, init_state = load_dataset()

    slots = ontology.slots
    model = load_model(args_save['model'], args_save, ontology, vocab)
    model.load_best_save(directory=args['dout'])
    model.initialize_state(init_state)
    if args['device'] is not None:
        model.cuda(args['device'])

    preds = model.run_pred(dataset[eva_set], args_save)
    eval_pred = dataset[eva_set].evaluate_preds(preds)
    with open(os.path.join(args['dout'], '%s_log.txt'%eva_set), 'wb') as f:
        f.write(str(eval_pred))
    pprint(eval_pred)
