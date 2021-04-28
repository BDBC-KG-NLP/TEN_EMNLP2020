import os

args = dict()

'''
configurations for basic model without label dependency
'''
# # root experiment folder
# args['dexp'] = 'TEN-XH'
# # dataset dir
# args['data_dir'] = 'woz'
# # args['data_dir'] = 'dstc2'
# args['data_dir'] = 'multi'
# # which model to use
# args['model'] = 'dst'
# # max epoch to run for
# args['epoch'] = 50
# # word embedding size
# args['demb'] = 400
# # hidden state size
# args['dhid'] = 50
# # tracker hidden state size
# args['hidden_s'] = 50 
# # batch size
# args['batch_size'] = 10
# # learning rate
# args['lr'] = 1e-3
# # slot to early stop on
# args['stop'] = 'joint_goal'
# # save directory to resume from
# args['resume'] = None
# # args['resume'] = 'exp/dst/base/epoch=45,iter=2760,train_joint_goal=0.991325,dev_joint_goal=0.901205.t7'
# # label dependency
# args['label_depend'] = False
# # submodel to use
# args['submodel'] = 'base'
# # args['submodel'] = 'state'
# # args['submodel'] = 'FGT'
# # random seed
# args['seed'] = 42
# # run in evaluation only mode
# args['test'] = False
# # which device to use
# args['device'] = 'cuda'
# # dropout rates
# args['dropout'] = {'emb': 0.2, 'att_in': 0.2, 'att_out': 0.2, 'tracker': 0.0}
# # output file
# args['dout'] = os.path.join(args['dexp'], args['model'], args['submodel'])
# if not os.path.isdir(args['dout']):
#     os.makedirs(args['dout'])



# '''
# configurations for basic model with label dependency
# '''
# root experiment folder
args['dexp'] = 'TEN-X'
# dataset dir
# args['data_dir'] = 'woz'
# args['data_dir'] = 'dstc2'
args['data_dir'] = 'multi'
# which model to use
args['model'] = 'dst'
# max epoch to run for
args['epoch'] = 100
# word embedding size
args['demb'] = 400
# hidden state size
args['dhid'] = 50
# tracker hidden state size
args['hidden_s'] = 50 
# batch size
args['batch_size'] = 20
# learning rate
args['lr'] = 1e-3
# slot to early stop on
args['stop'] = 'joint_goal'
# save directory to resume from
args['resume'] = None
# args['resume'] = 'exp/dst/base/epoch=45,iter=2760,train_joint_goal=0.991325,dev_joint_goal=0.901205.t7'
# label dependency
args['label_depend'] = True
# submodel to use
args['submodel'] = 'base'
# args['submodel'] = 'state'
# args['submodel'] = 'FGT'
# random seed
args['seed'] = 42
# run in evaluation only mode
args['test'] = False
# which device to use
args['device'] = 'cuda'
# dropout rates
args['dropout'] = {'emb': 0.2, 'att_in': 0.2, 'att_out': 0.2, 'tracker': 0.0}
# output file
args['dout'] = os.path.join(args['dexp'], args['model'], args['submodel'])
if not os.path.isdir(args['dout']):
    os.makedirs(args['dout'])



# '''
# configurations for model trained with states
# '''
# # root experiment folder
# args['dexp'] = 'TEN-Y'
# # dataset dir
# args['data_dir'] = 'woz'
# # args['data_dir'] = 'dstc2'
# args['data_dir'] = 'multi'
# # which model to use
# args['model'] = 'dst'
# # max epoch to run for
# args['epoch'] = 50
# # word embedding size
# args['demb'] = 400
# # hidden state size
# args['dhid'] = 50
# # tracker hidden state size
# args['hidden_s'] = 50 
# # batch size
# args['batch_size'] = 10
# # learning rate
# args['lr'] = 1e-3
# # slot to early stop on
# args['stop'] = 'joint_goal'
# # save directory to resume from
# args['resume'] = None
# # args['resume'] = 'exp/dst/base/epoch=45,iter=2760,train_joint_goal=0.991325,dev_joint_goal=0.901205.t7'
# # label dependency
# args['label_depend'] = True
# # submodel to use
# # args['submodel'] = 'base'
# args['submodel'] = 'state'
# # args['submodel'] = 'FGT'
# # random seed
# args['seed'] = 42
# # run in evaluation only mode
# args['test'] = False
# # which device to use
# args['device'] = 'cuda'
# # dropout rates
# args['dropout'] = {'emb': 0.2, 'att_in': 0.2, 'att_out': 0.2, 'tracker': 0.0}
# # output file
# args['dout'] = os.path.join(args['dexp'], args['model'], args['submodel'])
# if not os.path.isdir(args['dout']):
#     os.makedirs(args['dout'])



'''
Configurarions for FGT
'''
# # root experiment folder
# args['dexp'] = 'TEN'
# # dataset dir
# # args['data_dir'] = 'woz'
# # args['data_dir'] = 'dstc2'
# args['data_dir'] = 'multi'
# # which model to use
# args['model'] = 'dst'
# # max epoch to run for
# args['epoch'] = 50
# # word embedding size
# args['demb'] = 400
# # hidden state size
# args['dhid'] = 50
# # tracker hidden state size
# args['hidden_s'] = 50 
# # batch size
# args['batch_size'] = 120
# # learning rate
# args['lr'] = 1e-4
# # slot to early stop on
# args['stop'] = 'joint_goal'
# # save directory to resume from
# # args['resume'] = None
# args['resume'] = 'basic_depend/dst/base/epoch=47,iter=40416,dev_joint_goal=0.456876.t7'
# # label dependency
# args['label_depend'] = True
# # submodel to use
# # args['submodel'] = 'base'
# # args['submodel'] = 'state'
# args['submodel'] = 'FGT'
# # random seed
# args['seed'] = 42
# # run in evaluation only mode
# args['test'] = False
# # which device to use
# args['device'] = 'cuda'
# # dropout rates
# args['dropout'] = {'emb': 0.3, 'att_in': 0.3, 'att_out': 0.3, 'tracker': 0.0}
# # output file
# args['dout'] = os.path.join(args['dexp'], args['model'], args['submodel'])
# if not os.path.isdir(args['dout']):
#     os.makedirs(args['dout'])
