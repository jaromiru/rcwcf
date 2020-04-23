import torch

class Object:
	def init(self, args):
		#================== problem
		self.DATASET = args.dataset
		self.DATA_FILE = f"../data/{args.dataset}.json"
		self.META_FILE = f"../data/{args.dataset}_meta.json"

		#================== network
		self.BAG_SIZE = 64
		self.BATCH_SIZE = args.batch

		self.OPT_LR = 3.0e-3
		self.OPT_LR_FACTOR = 0.5

		self.OPT_L2 = 1.0e-4
		
		#================== iterations
		self.EPOCH_STEPS = args.eplen
		self.TRAINING_EPOCHS = args.epochs
		
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

config = Object()


