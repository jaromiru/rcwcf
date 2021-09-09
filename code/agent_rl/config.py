import torch

class Object:
	def init(self, args):
		#================== problem
		self.DATASET = args.dataset
		self.DATA_FILE = f"../data/{args.dataset}.json"
		self.META_FILE = f"../data/{args.dataset}_meta.json"

		self.MODEL_FILE = args.model

		#================== rl
		self.LAMBDA = args.target
		self.REWARD_CORRECT   = 1.
		self.REWARD_INCORRECT = 0.
		
		self.GAMMA  = 0.99
		
		#================== network
		self.BAG_SIZE = args.emb_size

		self.OPT_LR = args.lr
		self.OPT_LR_MIN = self.OPT_LR / 30

		self.OPT_L2 = args.l2
		self.OPT_MAX_NORM = 1.0
		self.OPT_LR_FACTOR = 0.5
		self.OPT_H_FACTOR = 1.0

		#================== training
		self.AGENTS = args.batch
		self.TRAINING_EPOCHS = args.epochs
		self.EPOCH_STEPS = args.eplen
		
		self.SAMPLE_CLS = args.sample_cls
		self.ALPHA_CLS = args.alpha_cls
		self.W_PI = 1.0
		self.W_V  = 0.5 
		self.W_H  = args.alpha_h
		self.W_CLS = 1.0

		self.LR_SCHEDULE = self.EPOCH_STEPS * 10
		self.H_SCHEDULE = self.EPOCH_STEPS
		self.H_MIN = args.alpha_h_min

		#================== log
		self.LOG_FILE = args.log

		#================== aux
		self.SEED = args.seed
		self.DATASEED = args.dataseed

		if args.device == 'auto':
			self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
		else:
			self.DEVICE = args.device

	def init_dataset(self, meta):
		self.CLASSES = meta['classes']
		self.LABEL = meta['label']
		self.DATA_DEPTH = meta['depth']
		self.TEST_SAMPLES = meta['test_samples']
		self.VAL_SAMPLES = meta['val_samples']

	def __str__(self):
		return str( vars(self) )

config = Object()


