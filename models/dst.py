import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
import logging
import os
import re
import json
from collections import defaultdict
from pprint import pformat

fix = {'centre': 'center', 'areas': 'area', 'phone number': 'number'}

def pad(seqs, emb, device, pad=0):
	lens = [len(s) for s in seqs]
	max_len = max(lens)
	padded = torch.LongTensor([s + (max_len-l) * [pad] for s, l in zip(seqs, lens)])
	return emb(padded.to(device)), lens

def asr_composing(asrs, asr_scores):
	composed = []
	lens = []
	for i in range(len(asr_scores)):        
		composed.append(torch.mean(asrs[i][0], 0))
		lens.append(max(asrs[i][1]))
	padded = nn.utils.rnn.pad_sequence(composed, batch_first=True, padding_value=0.)
	return padded, lens

def run_rnn(rnn, inputs, lens):
	# sort by lens
	order = np.argsort(lens)[::-1].tolist()
	reindexed = inputs.index_select(0, inputs.data.new(order).long())
	reindexed_lens = [lens[i] for i in order]
	packed = nn.utils.rnn.pack_padded_sequence(reindexed, reindexed_lens, batch_first=True)
	outputs, _ = rnn(packed)
	padded, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, padding_value=0.)
	reverse_order = np.argsort(order).tolist()
	recovered = padded.index_select(0, inputs.data.new(reverse_order).long())
	# reindexed_lens = [lens[i] for i in order]
	# recovered_lens = [reindexed_lens[i] for i in reverse_order]
	# assert recovered_lens == lens
	return recovered

def attend(seq, cond, lens):
	"""
	attend over the sequences `seq` using the condition `cond`.
	"""
	if len(cond.size())==1:
		scores = cond.expand_as(seq).mul(seq).sum(2)
	else:
		scores = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
	max_len = max(lens)
	for i, l in enumerate(lens):
		if l < max_len:
			scores.data[i, l:] = -np.inf
	weights = F.softmax(scores, dim=1).unsqueeze(2)
	context = weights.expand_as(seq).mul(seq).sum(1)
	return context

class FixedEmbedding(nn.Embedding):
	"""
	this is the same as `nn.Embedding` but detaches the result from the graph and has dropout after lookup.
	"""

	def __init__(self, dropout, *args, **kwargs):
		super(FixedEmbedding, self).__init__(*args, **kwargs)
		self.dropout = dropout

	def forward(self, *args, **kwargs):
		out = super(FixedEmbedding, self).forward(*args, **kwargs)
		out.detach_()
		return F.dropout(out, self.dropout, self.training)

class SlotAttention(nn.Module):
	"""
	slot attention component.
	"""

	def __init__(self, d_hid, dropout=0.):
		super(SlotAttention, self).__init__()
		self.slot_embedding = nn.Parameter(torch.Tensor(np.ones(d_hid)))
		nn.init.uniform_(self.slot_embedding, -1./d_hid, 1./d_hid)
		self.dropout = nn.Dropout(dropout)

	def forward(self, inp, lens):
		inp = self.dropout(inp)
		context = attend(inp, self.slot_embedding, lens)
		return context

class SeqEncoder(nn.Module):
	"""
	the sequence encoder.
	"""

	def __init__(self, din, dhid, slots, dropout=None):
		super(SeqEncoder, self).__init__()
		self.slots = slots        
		self.dropout = dropout or {}
		self.rnn_encoder = nn.GRU(din, dhid, bidirectional=True, batch_first=True)
		self.slot_att = nn.ModuleDict({s: SlotAttention(dhid*2, self.dropout.get('att_in', 0.2)) for s in self.slots})

	def forward(self, x, x_len, slot):
		rnn_encoding = run_rnn(self.rnn_encoder, x, x_len)
		attended = self.slot_att[slot](rnn_encoding, x_len)
		attended = F.dropout(attended, self.dropout.get('att_out', 0.2), self.training)
		return rnn_encoding, attended

class RNNTracker(nn.Module):
	"""
	the RNN tracker.
	"""

	def __init__(self, input_size, hidden_size, dropout=None):
		super(RNNTracker, self).__init__()
		self.dropout = dropout or {}
		self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
		# self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

	def forward(self, inputs):
		inputs = F.dropout(inputs, self.dropout.get('tracker', 0.0), self.training)
		inputs = inputs.unsqueeze(0)
		output, h_n = self.rnn(inputs) 
		output = output.squeeze(0)
		return output

class Model(nn.Module):
	"""
	the dialogue state tracking model.
	"""

	def __init__(self, args, ontology, vocab):
		super(Model, self).__init__()
		self.optimizer = None
		self.args = args
		self.device = args['device']     
		self.vocab = vocab
		self.ontology = ontology
		# self.slots = ['food', 'area', 'price range']
		self.slots = self.ontology.slots		
		self.gen_g()
		self.eos = self.vocab.word2index('<eos>')
		self.emb_fixed = FixedEmbedding(args['dropout'].get('emb', 0.2), len(vocab), args['demb'])    
		self.label_n = {s: len(self.ontology.values[s]) for s in self.ontology.values}
		hidden_s = args['hidden_s']
		self.utt_encoder = SeqEncoder(args['demb'], args['dhid'], self.slots, dropout=args['dropout'])
		self.act_encoder = SeqEncoder(args['demb'], args['dhid'], self.slots, dropout=args['dropout'])
		if args['label_depend']:			
			self.tracker = RNNTracker(args['dhid']*4, hidden_s, dropout=args['dropout'])
			self.classifier = nn.ModuleDict({s: nn.Linear(hidden_s, self.label_n[s]) for s in self.slots})
		else:
			self.classifier = nn.ModuleDict({s: nn.Linear(args['dhid']*4, self.label_n[s]) for s in self.slots})

	def initialize_state(self, state):
		self.init_state = dict()
		for s in self.slots:
			init_state = np.zeros(len(self.ontology.values[s]), dtype=np.float32)
			init_state[0] = 1.	
			self.init_state[s] = torch.as_tensor(init_state, device=self.device)
		
		# self.init_state = dict()
		# for s in self.slots:	
		# 	self.init_state[s] = torch.as_tensor(np.asarray(state[s], dtype=np.float32), device=self.device)

	def init_weights(self):
		for m in self.modules():
			if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
				for name, param in m.named_parameters():
					if 'weight_ih' in name:
						torch.nn.init.xavier_uniform_(param.data)
					elif 'weight_hh' in name:
						torch.nn.init.orthogonal_(param.data)
					elif 'bias' in name:
						param.data.fill_(0.)        

	def set_optimizer(self):
		self.optimizer = optim.Adam(self.parameters(), lr=self.args['lr'])

	def load_emb(self, Eword):
		new = self.emb_fixed.weight.data.new
		self.emb_fixed.weight.data.copy_(new(Eword))

	def gen_g(self):
		'''
		the function g(X1, T2, X2)
		'''
		self.g = dict()
		for s in self.slots:
			len_v = len(self.ontology.values[s])
			self.g[s] = np.zeros((len_v, len_v, len_v))
			for x2 in range(len_v):
				for x1 in range(len_v):
					for t1 in range(len_v):
						if t1==0 and x1==x2:
							self.g[s][x2, x1, t1] = 1.
						elif t1!=0 and t1==x2:
							self.g[s][x2, x1, t1] = 1.
		self.g = {s: torch.as_tensor(np.asarray(m, dtype=np.float32), device=self.device) for s, m in self.g.items()}

	def forward(self, batch):
		# convert to variables and look up embeddings  
		pad_ontology = {s: pad(v, self.emb_fixed, self.device, pad=self.eos) for s, v in self.ontology.num.items()}
		losses = [] 
		preds = {s:[] for s in self.slots}
		dia_states = {s:[] for s in self.slots}             
		dialogues = [[t for t in d.turns] for d in batch]
		turn_index = []

		for dialogue in dialogues:

			turn_index += range(len(dialogue))
			asrs = [pad(e.num['asr_transcripts'], self.emb_fixed, self.device, pad=self.eos) for e in dialogue]
			asr_scores = [torch.as_tensor(np.asarray(e.asr_scores, dtype=np.float32).reshape((-1, 1, 1)), device=self.device) for e in dialogue]
			utterance, utterance_len = asr_composing(asrs, asr_scores)

			acts = [pad(e.num['system_acts'], self.emb_fixed, self.device, pad=self.eos) for e in dialogue]

			labels = {s: [0 for i in range(len(dialogue))] for s in self.slots}
			for i, e in enumerate(dialogue):
				for s, v in e.turn_label:
					if s!='request':
						labels[s][i] = self.ontology.values[s].index(v)
			labels = {s: torch.as_tensor(np.asarray(m, dtype=np.int_), device=self.device) for s, m in labels.items()}

			states = {s: [0 for i in range(len(dialogue))] for s in self.slots}
			for i, e in enumerate(dialogue):
				for b in e.belief_state:
					s, v = b['slots'][0]
					s = fix.get(s.strip(), s.strip())
					v = fix.get(v.strip(), v.strip())
					if s!='slot':
						states[s][i] = self.ontology.values[s].index(v)
			states = {s: torch.as_tensor(np.asarray(m, dtype=np.int_), device=self.device) for s, m in states.items()}

			loss = 0.
			for s in self.slots:
				# for each slot, compute the scores for each value
				_, c_utt, = self.utt_encoder(utterance, utterance_len, slot=s)
				_, C_acts = list(zip(*[self.act_encoder(a, a_len, slot=s) for a, a_len in acts]))

				# compute the previous action score
				q_acts = []
				for i, C_act in enumerate(C_acts):
					q_act = attend(C_act.unsqueeze(0), c_utt[i].unsqueeze(0), lens=[C_act.size(0)])
					q_acts.append(q_act)
				c_acts = torch.cat(q_acts, dim=0)

				track_inputs = torch.cat([c_utt, c_acts], 1)
				if self.args['label_depend']:
					track_hidden = self.tracker(track_inputs)
				else:
					track_hidden = track_inputs
				output_s = self.classifier[s](track_hidden)

				if self.args['submodel']=='FGT':
					# train on factor graph					
					pred_t = F.softmax(output_s, 1)
					x = self.init_state[s]
					output_s = []
					for t in pred_t:
						uv = torch.matmul(x.unsqueeze(1), t.unsqueeze(0))
						x = torch.sum(self.g[s]*uv, (1,2))
						x = x/torch.sum(x)
						output_s.append(x.unsqueeze(0))
					pred_s = torch.cat(output_s, 0)
					output_s = torch.clamp(pred_s, min=1e-7, max=1.-1e-7)
					output_s = torch.log(output_s)				
					loss += F.nll_loss(output_s, states[s], reduction='none')
					preds[s].append(pred_s)

				elif self.args['submodel']=='base':
					# train with turn labels
					loss += F.cross_entropy(output_s, labels[s], reduction='none')
					pred = F.softmax(output_s, 1)
					preds[s].append(pred)

				else:
					# train with states
					loss += F.cross_entropy(output_s, states[s], reduction='none')
					pred = F.softmax(output_s, 1)
					preds[s].append(pred)					


				dia_states[s].extend(states[s])
			losses.append(loss)
			# losses.append(loss.unsqueeze(0))  
		all_loss = torch.mean(torch.cat(losses, 0))
		all_pred = {s: torch.cat(preds[s], 0) for s in self.slots}
		return all_loss, all_pred, dia_states, turn_index                              

	def get_train_logger(self):
		logger = logging.getLogger('train-{}'.format(self.__class__.__name__))
		formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
		file_handler = logging.FileHandler(os.path.join(self.args['dout'], 'train.log'))
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)
		return logger

	def run_train(self, train, dev, args):
		track = defaultdict(list)
		iteration = 0
		best = {}
		logger = self.get_train_logger()
		if self.optimizer is None:
			self.set_optimizer()

		for epoch in range(args['epoch']):
			logger.info('starting epoch {}'.format(epoch))

			# train and update parameters
			self.train()
			for batch in train.batch(batch_size=args['batch_size'], shuffle=True):
				iteration += 1
				self.zero_grad()
				loss, pred_all, dia_states, turn_index = self.forward(batch)
				loss.backward()
				# if self.args['clip']!=0:
				#     nn.utils.clip_grad_value_(self.parameters(), self.args['clip'])               
				self.optimizer.step()
				track['loss'].append(loss.item())

			# evalute on train and dev
			summary = {'iteration': iteration, 'epoch': epoch}
			for k, v in track.items():
				summary[k] = sum(v) / len(v)
			# summary.update({'eval_train_{}'.format(k): v for k, v in self.run_eval(train, args).items()})
			summary.update({'eval_dev_{}'.format(k): v for k, v in self.run_eval(dev, args).items()})

			# do early stopping saves
			stop_key = 'eval_dev_{}'.format(args['stop'])
			# train_key = 'eval_train_{}'.format(args['stop'])
			if best.get(stop_key, 0) <= summary[stop_key]:
				best_dev = '{:f}'.format(summary[stop_key])
				# best_train = '{:f}'.format(summary[train_key])
				best.update(summary)
				self.save(
					best,
					identifier='epoch={epoch},iter={iteration},dev_{key}={dev}'.format(
						epoch=epoch, iteration=iteration, dev=best_dev, key=args['stop'],
					)
				)
				self.prune_saves()
				# dev.record_preds(
				# 	preds=self.run_pred(dev, self.args),
				# 	to_file=os.path.join(self.args['dout'], 'dev.pred.json'),
				# )
			summary.update({'best_{}'.format(k): v for k, v in best.items()})
			logger.info(pformat(summary))
			track.clear() 

	def extract_predictions(self, pred_all, dia_states, turn_index):
		turn_size = len(turn_index)
		predictions = [set() for i in range(turn_size)]	

		if self.args['submodel']=='base':
			# Train with turn labels
			for s in self.slots:
				x = np.zeros((len(self.ontology.values[s])), dtype=np.float32)
				x[0] = 1.
				for i in range(turn_size):
					if turn_index[i]==0:
						x = np.zeros((len(self.ontology.values[s])), dtype=np.float32)
						x[0] = 1.
					p_t = pred_all[s][i]
					max_t = torch.argmax(p_t).item()
					if max_t!=0:
						state = np.zeros((len(self.ontology.values[s])), dtype=np.float32)
						state[max_t] = 1.
					else:
						state = x              	
					pred_index = np.argmax(state)
					# with open('pred_%s.txt'%s, 'ab') as f:
					# 	f.write('\t'.join([str(dia_states[s][i]), str(pred_index), str(state.tolist())]) +'\n')				
					if pred_index!=0:
						predictions[i].add((s, self.ontology.values[s][pred_index]))
					x = state
					
			# # Output turn labels
			# for s in self.slots:
			# 	for i in range(turn_size):
			# 		x = pred_all[s][i]				
			# 		pred_index = torch.argmax(x).item()			
			# 		if pred_index!=0:
			# 			predictions[i].add((s, self.ontology.values[s][pred_index]))					

		elif self.args['submodel']=='FGT':
			# Train and test on factor graphs
			for s in self.slots:
				for i in range(turn_size):
					x = pred_all[s][i]				
					pred_index = torch.argmax(x).item()			
					if pred_index!=0:
						predictions[i].add((s, self.ontology.values[s][pred_index]))					

		else:
			# Train and test with states
			for s in self.slots:
				for i in range(turn_size):
					x = pred_all[s][i]				
					pred_index = torch.argmax(x).item()			
					if pred_index!=0:
						predictions[i].add((s, self.ontology.values[s][pred_index]))

		return predictions		     

	def run_pred(self, dev, args):
		self.eval()
		predictions = []
		for batch in dev.batch(batch_size=args['batch_size']):
			loss, pred_all, pred_states, turn_index = self.forward(batch)
			predictions += self.extract_predictions(pred_all, pred_states, turn_index)
		return predictions

	def run_eval(self, dev, args):
		predictions = self.run_pred(dev, args)
		return dev.evaluate_preds(predictions)

	def save_config(self):
		fname = '{}/config.json'.format(self.args['dout'])
		with open(fname, 'wt') as f:
			logging.info('saving config to {}'.format(fname))
			json.dump(self.args, f, indent=2)

	@classmethod
	def load_config(cls, fname, ontology, **kwargs):
		with open(fname) as f:
			logging.info('loading config from {}'.format(fname))
			args = object()
			for k, v in json.load(f):
				setattr(args, k, kwargs.get(k, v))
		return cls(args, ontology)

	def save(self, summary, identifier):
		fname = '{}/{}.t7'.format(self.args['dout'], identifier)
		logging.info('saving model to {}'.format(fname))
		state = {
			'args': self.args,
			'model': self.state_dict(),
			'summary': summary,
			'optimizer': self.optimizer.state_dict(),
		}
		torch.save(state, fname)

	def load(self, fname):
		logging.info('loading model from {}'.format(fname))
		state = torch.load(fname)
		self.load_state_dict(state['model'])
		self.set_optimizer()
		self.optimizer.load_state_dict(state['optimizer'])

	def get_saves(self, directory=None):
		if directory is None:
			directory = self.args['dout']
		files = [f for f in os.listdir(directory) if f.endswith('.t7')]
		scores = []
		for fname in files:
			re_str = r'dev_{}=([0-9\.]+)'.format(self.args['stop'])
			dev_acc = re.findall(re_str, fname)
			if dev_acc:
				score = float(dev_acc[0].strip('.'))
				scores.append((score, os.path.join(directory, fname)))
		if not scores:
			raise Exception('No files found!')
		scores.sort(key=lambda tup: tup[0], reverse=True)
		return scores

	def prune_saves(self, n_keep=5):
		scores_and_files = self.get_saves()
		if len(scores_and_files) > n_keep:
			for score, fname in scores_and_files[n_keep:]:
				os.remove(fname)

	def load_best_save(self, directory):
		if directory is None:
			directory = self.args['dout']

		scores_and_files = self.get_saves(directory=directory)
		if scores_and_files:
			assert scores_and_files, 'no saves exist at {}'.format(directory)
			score, fname = scores_and_files[0]
			self.load(fname)
