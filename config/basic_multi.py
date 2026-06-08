import torchvision.transforms as transforms
from PIL import Image

class ConfigBasic:
    def __init__(self,):
        self.dataset = None
        self.setting = None
        self.logscale = False
        self.set_optimizer_parameters()
        self.set_training_opts()
        self.set_network()

    def set_dataset(self):
        if 'morph' in self.dataset:
            if self.logscale:
                self.tau = 0.1
            else:
                self.tau = 2

            self.morph_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/MORPH'
            self.morph_setting = 'A'
            self.morph_fold = 0
            self.clap_fold = 'eval_on_test'
            if self.morph_setting == 'A':
                self.morph_is_filelist = True
                self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingA/SettingA_fold{self.morph_fold}_train.txt'
                self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingA/SettingA_fold{self.morph_fold}_test.txt'
                self.morph_delimeter = ","
                self.morph_img_idx = 4
                self.morph_lb_idx = 3

            elif self.morph_setting == 'B':
                self.morph_delimeter = " "
                self.morph_img_idx = 3
                self.morph_lb_idx = 2
                if self.morph_fold == 1:
                    self.morph_is_filelist = True
                    self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S1_train.txt'
                    self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S2+S3_test.txt'
                else:
                    self.morph_is_filelist = True
                    self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S2_train.txt'
                    self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S1+S3_test.txt'

            elif self.setting == 'C':
                self.morph_delimeter = " "
                self.morph_img_idx = 0
                self.morph_lb_idx = 2
                self.morph_is_filelist = True
                self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingC/SettingC_fold{self.morph_fold}_train.txt'
                self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingC/SettingC_fold{self.morph_fold}_test.txt'

            elif self.morph_setting == 'D':
                self.morph_delimeter = " "
                self.morph_img_idx = 0
                self.morph_lb_idx = 2
                self.morph_is_filelist = True
                self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingD/SettingD_fold{self.morph_fold}_train.txt'
                self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingD/SettingD_fold{self.morph_fold}_test.txt'
            else:
                raise ValueError(f'MORPH setting {self.morph_setting} is out of range.')

        if 'adience' in self.dataset:
            # self.adience_is_filelist = False
            self.adience_fold = 0
            self.adience_is_filelist = True
            self.adience_train_file = f'/hdd1/cwlee/datasets/OrderLearning/Adience/adience_F{self.adience_fold}_train_algn_[0_7]_v2.pickle'
            self.adience_test_file = f'/hdd1/cwlee/datasets/OrderLearning/Adience/adience_F{self.adience_fold}_test_algn_[0_7]_v2.pickle'

        if 'clap' in self.dataset:
            self.clap_delimeter = " "
            self.clap_img_idx = 0
            self.clap_lb_idx = 1
            self.clap_is_filelist = True
            self.clap_fold = 'eval_on_test'
            self.clap_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/CLAP/2015'
            if self.clap_fold == 'eval_on_test':
                self.clap_train_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_trainval.txt'
                self.clap_test_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_test.txt'
            elif self.clap_fold == 'eval_on_val':
                self.clap_train_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_train.txt'
                self.clap_test_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_val.txt'
            else:
                raise ValueError(f'check fold: it should be [eval_on_test] or [eval_on_val], but {self.clap_fold} is given.')

        if 'ageDB' in self.dataset:
            self.ageDB_is_filelist = True
            self.ageDB_img_root = '/hdd1/cwlee/datasets/OrderLearning/img'
            self.ageDB_data_file = '/hdd1/cwlee/datasets/OrderLearning/index/AgeDB/agedb.csv'
        if 'utk' in self.dataset:
            self.utk_is_filelist = True
            self.utk_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/utk'
            self.utk_train_file = '/hdd1/cwlee/datasets/OrderLearning/index/utk/UTK_train_coral.csv'
            self.utk_test_file = '/hdd1/cwlee/datasets/OrderLearning/index/utk/UTK_test_coral.csv'

        if 'cacd' in self.dataset:
            self.cacd_is_filelist = True
            self.cacd_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/CACD'
            self.cacd_data_file = '/hdd1/cwlee/datasets/OrderLearning/index/cacd/CACD.csv'

        self.mean= [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]

        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)


        self.transform_tr = transforms.Compose([
                                               lambda x: Image.fromarray(x).convert('RGB'),
                                               transforms.RandomResizedCrop(size=224, scale=(0.2, 1.), interpolation=3, antialias=True),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.RandomApply([
                                                   transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                               ], p=0.8),
                                               transforms.RandomGrayscale(p=0.2),
                                               self.normalize
                                               ])
        
        self.transform_te = transforms.Compose([
                                               lambda x: Image.fromarray(x).convert('RGB'),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               self.normalize
                                               ])

    def set_optimizer_parameters(self):
        # *** Optimizer
        self.adam = True
        self.learning_rate = 0.0001
        self.lr_decay_epochs = [30, 50, 100]
        self.lr_decay_rate = 0.1
        self.momentum = 0.9
        self.weight_decay = 0.0005

        # *** Scheduler
        self.scheduler = 'cosine'

    def set_network(self):
        self.model = 'T_v0'
        self.backbone = 'vitB16'
        self.ckpt = None

    def set_training_opts(self):
        # *** Print Option
        self.val_freq = 3
        self.print_freq = 50

        # *** Training
        self.batch_size = 32
        self.num_workers = 1
        self.epochs = 100

        # *** Save option
        self.save_freq = 100
        self.wandb = False

    def set_test_opts(self):
        self.ckpt = None


class ConfigBasic_fold1:
    def __init__(self,):
        self.dataset = None
        self.setting = None
        self.logscale = False
        self.set_optimizer_parameters()
        self.set_training_opts()
        self.set_network()

    def set_dataset(self):
        if 'morph' in self.dataset:
            if self.logscale:
                self.tau = 0.1
            else:
                self.tau = 2

            self.morph_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/MORPH'
            self.morph_setting = 'A'
            self.morph_fold = 1
            self.clap_fold = 'eval_on_test'
            if self.morph_setting == 'A':
                self.morph_is_filelist = True
                self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingA/SettingA_fold{self.morph_fold}_train.txt'
                self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingA/SettingA_fold{self.morph_fold}_test.txt'
                self.morph_delimeter = ","
                self.morph_img_idx = 4
                self.morph_lb_idx = 3

            elif self.morph_setting == 'B':
                self.morph_delimeter = " "
                self.morph_img_idx = 3
                self.morph_lb_idx = 2
                if self.morph_fold == 1:
                    self.morph_is_filelist = True
                    self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S1_train.txt'
                    self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S2+S3_test.txt'
                else:
                    self.morph_is_filelist = True
                    self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S2_train.txt'
                    self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S1+S3_test.txt'

            elif self.setting == 'C':
                self.morph_delimeter = " "
                self.morph_img_idx = 0
                self.morph_lb_idx = 2
                self.morph_is_filelist = True
                self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingC/SettingC_fold{self.morph_fold}_train.txt'
                self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingC/SettingC_fold{self.morph_fold}_test.txt'

            elif self.morph_setting == 'D':
                self.morph_delimeter = " "
                self.morph_img_idx = 0
                self.morph_lb_idx = 2
                self.morph_is_filelist = True
                self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingD/SettingD_fold{self.morph_fold}_train.txt'
                self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingD/SettingD_fold{self.morph_fold}_test.txt'
            else:
                raise ValueError(f'MORPH setting {self.morph_setting} is out of range.')
            

        if 'adience' in self.dataset:
            # self.adience_is_filelist = False
            self.adience_fold = 1
            self.adience_is_filelist = True
            self.adience_train_file = f'/hdd1/cwlee/datasets/OrderLearning/Adience/adience_F{self.adience_fold}_train_algn_[0_7]_v2.pickle'
            self.adience_test_file = f'/hdd1/cwlee/datasets/OrderLearning/Adience/adience_F{self.adience_fold}_test_algn_[0_7]_v2.pickle'

        if 'clap' in self.dataset:
            self.clap_delimeter = " "
            self.clap_img_idx = 0
            self.clap_lb_idx = 1
            self.clap_is_filelist = True
            self.clap_fold = 'eval_on_test'
            self.clap_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/CLAP/2015'
            if self.clap_fold == 'eval_on_test':
                self.clap_train_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_trainval.txt'
                self.clap_test_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_test.txt'
            elif self.clap_fold == 'eval_on_val':
                self.clap_train_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_train.txt'
                self.clap_test_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_val.txt'
            else:
                raise ValueError(f'check fold: it should be [eval_on_test] or [eval_on_val], but {self.clap_fold} is given.')

        if 'ageDB' in self.dataset:
            self.ageDB_is_filelist = True
            self.ageDB_img_root = '/hdd1/cwlee/datasets/OrderLearning/img'
            self.ageDB_data_file = '/hdd1/cwlee/datasets/OrderLearning/index/AgeDB/agedb.csv'

        if 'utk' in self.dataset:
            self.utk_is_filelist = True
            self.utk_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/utk'
            self.utk_train_file = '/hdd1/cwlee/datasets/OrderLearning/index/utk/UTK_train_coral.csv'
            self.utk_test_file = '/hdd1/cwlee/datasets/OrderLearning/index/utk/UTK_test_coral.csv'

        if 'cacd' in self.dataset:
            self.cacd_is_filelist = True
            self.cacd_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/CACD'
            self.cacd_data_file = '/hdd1/cwlee/datasets/OrderLearning/index/cacd/CACD.csv'

        self.mean= [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]

        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.transform_tr = transforms.Compose([
                                               lambda x: Image.fromarray(x).convert('RGB'),
                                               transforms.RandomResizedCrop(size=224, scale=(0.2, 1.), interpolation=3, antialias=True),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.RandomApply([
                                                   transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                               ], p=0.8),
                                               transforms.RandomGrayscale(p=0.2),
                                               self.normalize
                                               ])
        
        self.transform_te = transforms.Compose([
                                               lambda x: Image.fromarray(x).convert('RGB'),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               self.normalize
                                               ])

    def set_optimizer_parameters(self):
        # *** Optimizer
        self.adam = True
        self.learning_rate = 0.0001
        self.lr_decay_epochs = [30, 50, 100]
        self.lr_decay_rate = 0.1
        self.momentum = 0.9
        self.weight_decay = 0.0005

        # *** Scheduler
        self.scheduler = 'cosine'

    def set_network(self):
        self.model = 'T_v0'
        self.backbone = 'vitB16'
        self.ckpt = None

    def set_training_opts(self):
        # *** Print Option
        self.val_freq = 3
        self.print_freq = 50

        # *** Training
        self.batch_size = 32
        self.num_workers = 1
        self.epochs = 100

        # *** Save option
        self.save_freq = 100
        self.wandb = False

    def set_test_opts(self):
        self.ckpt = None

class ConfigBasic_fold2:
    def __init__(self,):
        self.dataset = None
        self.setting = None
        self.logscale = False
        self.set_optimizer_parameters()
        self.set_training_opts()
        self.set_network()

    def set_dataset(self):
        if 'morph' in self.dataset:
            if self.logscale:
                self.tau = 0.1
            else:
                self.tau = 2

            self.morph_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/MORPH'
            self.morph_setting = 'A'
            self.morph_fold = 2
            self.clap_fold = 'eval_on_test'
            if self.morph_setting == 'A':
                self.morph_is_filelist = True
                self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingA/SettingA_fold{self.morph_fold}_train.txt'
                self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingA/SettingA_fold{self.morph_fold}_test.txt'
                self.morph_delimeter = ","
                self.morph_img_idx = 4
                self.morph_lb_idx = 3

            elif self.morph_setting == 'B':
                self.morph_delimeter = " "
                self.morph_img_idx = 3
                self.morph_lb_idx = 2
                if self.morph_fold == 1:
                    self.morph_is_filelist = True
                    self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S1_train.txt'
                    self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S2+S3_test.txt'
                else:
                    self.morph_is_filelist = True
                    self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S2_train.txt'
                    self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S1+S3_test.txt'

            elif self.setting == 'C':
                self.morph_delimeter = " "
                self.morph_img_idx = 0
                self.morph_lb_idx = 2
                self.morph_is_filelist = True
                self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingC/SettingC_fold{self.morph_fold}_train.txt'
                self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingC/SettingC_fold{self.morph_fold}_test.txt'

            elif self.morph_setting == 'D':
                self.morph_delimeter = " "
                self.morph_img_idx = 0
                self.morph_lb_idx = 2
                self.morph_is_filelist = True
                self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingD/SettingD_fold{self.morph_fold}_train.txt'
                self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingD/SettingD_fold{self.morph_fold}_test.txt'
            else:
                raise ValueError(f'MORPH setting {self.morph_setting} is out of range.')

        if 'adience' in self.dataset:
            # self.adience_is_filelist = False
            self.adience_fold = 2
            self.adience_is_filelist = True
            self.adience_train_file = f'/hdd1/cwlee/datasets/OrderLearning/Adience/adience_F{self.adience_fold}_train_algn_[0_7]_v2.pickle'
            self.adience_test_file = f'/hdd1/cwlee/datasets/OrderLearning/Adience/adience_F{self.adience_fold}_test_algn_[0_7]_v2.pickle'
            # self.adience_tau = 1

        if 'clap' in self.dataset:
            self.clap_delimeter = " "
            self.clap_img_idx = 0
            self.clap_lb_idx = 1
            self.clap_is_filelist = True
            self.clap_fold = 'eval_on_test'
            self.clap_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/CLAP/2015'
            if self.clap_fold == 'eval_on_test':
                self.clap_train_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_trainval.txt'
                self.clap_test_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_test.txt'
            elif self.clap_fold == 'eval_on_val':
                self.clap_train_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_train.txt'
                self.clap_test_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_val.txt'
            else:
                raise ValueError(f'check fold: it should be [eval_on_test] or [eval_on_val], but {self.clap_fold} is given.')

        if 'ageDB' in self.dataset:
            self.ageDB_is_filelist = True
            self.ageDB_img_root = '/hdd1/cwlee/datasets/OrderLearning/img'
            self.ageDB_data_file = '/hdd1/cwlee/datasets/OrderLearning/index/AgeDB/agedb.csv'
        if 'utk' in self.dataset:
            self.utk_is_filelist = True
            self.utk_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/utk'
            self.utk_train_file = '/hdd1/cwlee/datasets/OrderLearning/index/utk/UTK_train_coral.csv'
            self.utk_test_file = '/hdd1/cwlee/datasets/OrderLearning/index/utk/UTK_test_coral.csv'

        if 'cacd' in self.dataset:
            self.cacd_is_filelist = True
            self.cacd_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/CACD'
            self.cacd_data_file = '/hdd1/cwlee/datasets/OrderLearning/index/cacd/CACD.csv'
        
        if 'imdb-wiki' in self.dataset:
            self.imdb_is_filelist = True
            self.imdb_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/IMDB_WIKI'
            self.imdb_file = '/hdd1/cwlee/datasets/OrderLearning/img/IMDB_WIKI/imdb_aligned_with_facial_points.txt'
            self.wiki_file = '/hdd1/cwlee/datasets/OrderLearning/img/IMDB_WIKI/UTK_test_coral.csv'
       
        # else:
        #     raise ValueError(f'{self.dataset} is out of range!')

        self.mean= [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]

        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.transform_tr = transforms.Compose([
                                               lambda x: Image.fromarray(x).convert('RGB'),
                                               transforms.RandomResizedCrop(size=224, scale=(0.2, 1.), interpolation=3, antialias=True),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.RandomApply([
                                                   transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                               ], p=0.8),
                                               transforms.RandomGrayscale(p=0.2),
                                               self.normalize
                                               ])
        
        self.transform_te = transforms.Compose([
                                               lambda x: Image.fromarray(x).convert('RGB'),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               self.normalize
                                               ])
   
    def set_optimizer_parameters(self):
        # *** Optimizer
        self.adam = True
        self.learning_rate = 0.0001
        self.lr_decay_epochs = [30, 50, 100]
        self.lr_decay_rate = 0.1
        self.momentum = 0.9
        self.weight_decay = 0.0005

        # *** Scheduler
        self.scheduler = 'cosine'

    def set_network(self):
        self.model = 'T_v0'
        # self.backbone = 'vgg16bn'
        self.backbone = 'vitB16'
        self.ckpt = None

    def set_training_opts(self):
        # *** Print Option
        self.val_freq = 3
        self.print_freq = 50

        # *** Training
        self.batch_size = 32
        self.num_workers = 1
        self.epochs = 100

        # *** Save option
        self.save_freq = 100
        self.wandb = False

    def set_test_opts(self):
        self.ckpt = None

class ConfigBasic_fold3:
    def __init__(self,):
        self.dataset = None
        self.setting = None
        self.logscale = False
        self.set_optimizer_parameters()
        self.set_training_opts()
        self.set_network()

    def set_dataset(self):
        if 'morph' in self.dataset:
            if self.logscale:
                self.tau = 0.1
            else:
                self.tau = 2

            self.morph_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/MORPH'
            self.morph_setting = 'A'
            self.morph_fold = 3
            self.clap_fold = 'eval_on_test'
            if self.morph_setting == 'A':
                self.morph_is_filelist = True
                self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingA/SettingA_fold{self.morph_fold}_train.txt'
                self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingA/SettingA_fold{self.morph_fold}_test.txt'
                self.morph_delimeter = ","
                self.morph_img_idx = 4
                self.morph_lb_idx = 3

            elif self.morph_setting == 'B':
                self.morph_delimeter = " "
                self.morph_img_idx = 3
                self.morph_lb_idx = 2
                if self.morph_fold == 1:
                    self.morph_is_filelist = True
                    self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S1_train.txt'
                    self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S2+S3_test.txt'
                else:
                    self.morph_is_filelist = True
                    self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S2_train.txt'
                    self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S1+S3_test.txt'

            elif self.setting == 'C':
                self.morph_delimeter = " "
                self.morph_img_idx = 0
                self.morph_lb_idx = 2
                self.morph_is_filelist = True
                self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingC/SettingC_fold{self.morph_fold}_train.txt'
                self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingC/SettingC_fold{self.morph_fold}_test.txt'

            elif self.morph_setting == 'D':
                self.morph_delimeter = " "
                self.morph_img_idx = 0
                self.morph_lb_idx = 2
                self.morph_is_filelist = True
                self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingD/SettingD_fold{self.morph_fold}_train.txt'
                self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingD/SettingD_fold{self.morph_fold}_test.txt'
            else:
                raise ValueError(f'MORPH setting {self.morph_setting} is out of range.')

        if 'morphC' in self.dataset:
            if self.logscale:
                self.tau = 0.1
            else:
                self.tau = 2

            self.morph_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/MORPH'
            self.morph_setting = 'C'
            self.morph_fold = 0
            self.clap_fold = 'eval_on_test'
      
            self.morph_delimeter = " "
            self.morph_img_idx = 0
            self.morph_lb_idx = 2
            self.morph_is_filelist = True
            self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingC/setting_C_train_fold{self.morph_fold}.txt'
            self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingC/setting_C_test_fold{self.morph_fold}.txt'

        if 'morphD' in self.dataset:
            if self.logscale:
                self.tau = 0.1
            else:
                self.tau = 2

            self.morph_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/MORPH'
            self.morph_setting = 'D'
            self.morph_fold = 0
            self.clap_fold = 'eval_on_test'
      
            self.morph_delimeter = " "
            self.morph_img_idx = 0
            self.morph_lb_idx = 2
            self.morph_is_filelist = True
            self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingD/setting_D_train_fold{self.morph_fold}.txt'
            self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingD/setting_D_test_fold{self.morph_fold}.txt'

        if 'adience' in self.dataset:
            # self.adience_is_filelist = False
            self.adience_fold = 3
            self.adience_is_filelist = True
            self.adience_train_file = f'/hdd1/cwlee/datasets/OrderLearning/Adience/adience_F{self.adience_fold}_train_algn_[0_7]_v2.pickle'
            self.adience_test_file = f'/hdd1/cwlee/datasets/OrderLearning/Adience/adience_F{self.adience_fold}_test_algn_[0_7]_v2.pickle'
            # self.adience_tau = 1

        if 'clap' in self.dataset:
            self.clap_delimeter = " "
            self.clap_img_idx = 0
            self.clap_lb_idx = 1
            self.clap_is_filelist = True
            self.clap_fold = 'eval_on_test'
            self.clap_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/CLAP/2015'
            if self.clap_fold == 'eval_on_test':
                # self.clap_train_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_trainval.txt'
                # self.clap_test_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_test.txt'
                self.clap_train_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_trainval.txt'
                self.clap_test_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split_demographics/CLAP_test.txt'
            elif self.clap_fold == 'eval_on_val':
                self.clap_train_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_train.txt'
                self.clap_test_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_val.txt'
            else:
                raise ValueError(f'check fold: it should be [eval_on_test] or [eval_on_val], but {self.clap_fold} is given.')

        if 'ageDB' in self.dataset:
            self.ageDB_is_filelist = True
            self.ageDB_img_root = '/hdd1/cwlee/datasets/OrderLearning/img'
            # self.ageDB_data_file = '/hdd1/cwlee/datasets/OrderLearning/index/AgeDB/agedb.csv'
            self.ageDB_data_file = '/hdd1/cwlee/datasets/OrderLearning/index/AgeDB/agedb_race.csv'

        if 'utk' in self.dataset:
            self.utk_is_filelist = True
            self.utk_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/utk'
            self.utk_train_file = '/hdd1/cwlee/datasets/OrderLearning/index/utk/UTK_train_coral.csv'
            self.utk_test_file = '/hdd1/cwlee/datasets/OrderLearning/index/utk/UTK_test_coral.csv'

        if 'cacd' in self.dataset:
            self.cacd_is_filelist = True
            self.cacd_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/CACD'
            # self.cacd_data_file = '/hdd1/cwlee/datasets/OrderLearning/index/cacd/CACD.csv'
            self.cacd_data_file = '/hdd1/cwlee/datasets/OrderLearning/index/cacd/CACD_race.csv'
        
        if 'imdb-wiki' in self.dataset:
            self.imdb_is_filelist = True
            self.imdb_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/IMDB_WIKI'
            self.imdb_file = '/hdd1/cwlee/datasets/OrderLearning/img/IMDB_WIKI/imdb_aligned_with_facial_points.txt'
            self.wiki_file = '/hdd1/cwlee/datasets/OrderLearning/img/IMDB_WIKI/UTK_test_coral.csv'
       
        # else:
        #     raise ValueError(f'{self.dataset} is out of range!')

        self.mean= [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]

        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.transform_tr = transforms.Compose([
                                               lambda x: Image.fromarray(x).convert('RGB'),
                                               transforms.RandomResizedCrop(size=224, scale=(0.2, 1.), interpolation=3, antialias=True),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.RandomApply([
                                                   transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                               ], p=0.8),
                                               transforms.RandomGrayscale(p=0.2),
                                               self.normalize
                                               ])
        
        self.transform_te = transforms.Compose([
                                               lambda x: Image.fromarray(x).convert('RGB'),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               self.normalize
                                               ])
    def set_optimizer_parameters(self):
        # *** Optimizer
        self.adam = True
        self.learning_rate = 0.0001
        self.lr_decay_epochs = [30, 50, 100]
        self.lr_decay_rate = 0.1
        self.momentum = 0.9
        self.weight_decay = 0.0005

        # *** Scheduler
        self.scheduler = 'cosine'

    def set_network(self):
        self.model = 'T_v0'
        # self.backbone = 'vgg16bn'
        self.backbone = 'vitB16'
        self.ckpt = None

    def set_training_opts(self):
        # *** Print Option
        self.val_freq = 3
        self.print_freq = 50

        # *** Training
        self.batch_size = 32
        self.num_workers = 1
        self.epochs = 100

        # *** Save option
        self.save_freq = 100
        self.wandb = False

    def set_test_opts(self):
        self.ckpt = None

class ConfigBasic_fold4:
    def __init__(self,):
        self.dataset = None
        self.setting = None
        self.logscale = False
        self.set_optimizer_parameters()
        self.set_training_opts()
        self.set_network()

    def set_dataset(self):
        if 'morph' in self.dataset:
            if self.logscale:
                self.tau = 0.1
            else:
                self.tau = 2

            self.morph_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/MORPH'
            self.morph_setting = 'A'
            self.morph_fold = 4
            self.clap_fold = 'eval_on_test'
            if self.morph_setting == 'A':
                self.morph_is_filelist = True
                self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingA/SettingA_fold{self.morph_fold}_train.txt'
                self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingA/SettingA_fold{self.morph_fold}_test.txt'
                self.morph_delimeter = ","
                self.morph_img_idx = 4
                self.morph_lb_idx = 3

            elif self.morph_setting == 'B':
                self.morph_delimeter = " "
                self.morph_img_idx = 3
                self.morph_lb_idx = 2
                if self.morph_fold == 1:
                    self.morph_is_filelist = True
                    self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S1_train.txt'
                    self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S2+S3_test.txt'
                else:
                    self.morph_is_filelist = True
                    self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S2_train.txt'
                    self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingB/SettingB_S1+S3_test.txt'

            elif self.setting == 'C':
                self.morph_delimeter = " "
                self.morph_img_idx = 0
                self.morph_lb_idx = 2
                self.morph_is_filelist = True
                self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingC/SettingC_fold{self.morph_fold}_train.txt'
                self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingC/SettingC_fold{self.morph_fold}_test.txt'

            elif self.morph_setting == 'D':
                self.morph_delimeter = " "
                self.morph_img_idx = 0
                self.morph_lb_idx = 2
                self.morph_is_filelist = True
                self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingD/SettingD_fold{self.morph_fold}_train.txt'
                self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingD/SettingD_fold{self.morph_fold}_test.txt'
            else:
                raise ValueError(f'MORPH setting {self.morph_setting} is out of range.')

        if 'morphC' in self.dataset:
            if self.logscale:
                self.tau = 0.1
            else:
                self.tau = 2

            self.morph_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/MORPH'
            self.morph_setting = 'C'
            self.morph_fold = 0
            self.clap_fold = 'eval_on_test'
      
            self.morph_delimeter = " "
            self.morph_img_idx = 0
            self.morph_lb_idx = 2
            self.morph_is_filelist = True
            self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingC/setting_C_train_fold{self.morph_fold}.txt'
            self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingC/setting_C_test_fold{self.morph_fold}.txt'

        if 'morphD' in self.dataset:
            if self.logscale:
                self.tau = 0.1
            else:
                self.tau = 2

            self.morph_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/MORPH'
            self.morph_setting = 'D'
            self.morph_fold = 0
            self.clap_fold = 'eval_on_test'
      
            self.morph_delimeter = " "
            self.morph_img_idx = 0
            self.morph_lb_idx = 2
            self.morph_is_filelist = True
            self.morph_train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingD/setting_D_train_fold{self.morph_fold}.txt'
            self.morph_test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_SettingD/setting_D_test_fold{self.morph_fold}.txt'

        if 'adience' in self.dataset:
            # self.adience_is_filelist = False
            self.adience_fold = 4
            self.adience_is_filelist = True
            self.adience_train_file = f'/hdd1/cwlee/datasets/OrderLearning/Adience/adience_F{self.adience_fold}_train_algn_[0_7]_v2.pickle'
            self.adience_test_file = f'/hdd1/cwlee/datasets/OrderLearning/Adience/adience_F{self.adience_fold}_test_algn_[0_7]_v2.pickle'
            # self.adience_tau = 1

        if 'clap' in self.dataset:
            self.clap_delimeter = " "
            self.clap_img_idx = 0
            self.clap_lb_idx = 1
            self.clap_is_filelist = True
            self.clap_fold = 'eval_on_test'
            self.clap_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/CLAP/2015'
            if self.clap_fold == 'eval_on_test':
                # self.clap_train_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_trainval.txt'
                # self.clap_test_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_test.txt'
                self.clap_train_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_trainval.txt'
                self.clap_test_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split_demographics/CLAP_test.txt'
            elif self.clap_fold == 'eval_on_val':
                self.clap_train_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_train.txt'
                self.clap_test_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_val.txt'
            else:
                raise ValueError(f'check fold: it should be [eval_on_test] or [eval_on_val], but {self.clap_fold} is given.')

        if 'ageDB' in self.dataset:
            self.ageDB_is_filelist = True
            self.ageDB_img_root = '/hdd1/cwlee/datasets/OrderLearning/img'
            # self.ageDB_data_file = '/hdd1/cwlee/datasets/OrderLearning/index/AgeDB/agedb.csv'
            self.ageDB_data_file = '/hdd1/cwlee/datasets/OrderLearning/index/AgeDB/agedb_race.csv'

        if 'utk' in self.dataset:
            self.utk_is_filelist = True
            self.utk_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/utk'
            self.utk_train_file = '/hdd1/cwlee/datasets/OrderLearning/index/utk/UTK_train_coral.csv'
            self.utk_test_file = '/hdd1/cwlee/datasets/OrderLearning/index/utk/UTK_test_coral.csv'

        if 'cacd' in self.dataset:
            self.cacd_is_filelist = True
            self.cacd_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/CACD'
            # self.cacd_data_file = '/hdd1/cwlee/datasets/OrderLearning/index/cacd/CACD.csv'

            self.cacd_data_file = '/hdd1/cwlee/datasets/OrderLearning/index/cacd/CACD_race.csv'
        
        if 'imdb-wiki' in self.dataset:
            self.imdb_is_filelist = True
            self.imdb_img_root = '/hdd1/cwlee/datasets/OrderLearning/img/IMDB_WIKI'
            self.imdb_file = '/hdd1/cwlee/datasets/OrderLearning/img/IMDB_WIKI/imdb_aligned_with_facial_points.txt'
            self.wiki_file = '/hdd1/cwlee/datasets/OrderLearning/img/IMDB_WIKI/UTK_test_coral.csv'
       
        # else:
        #     raise ValueError(f'{self.dataset} is out of range!')

        self.mean= [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]

        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.transform_tr = transforms.Compose([
                                               lambda x: Image.fromarray(x).convert('RGB'),
                                               transforms.RandomResizedCrop(size=224, scale=(0.2, 1.), interpolation=3, antialias=True),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.RandomApply([
                                                   transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                               ], p=0.8),
                                               transforms.RandomGrayscale(p=0.2),
                                               self.normalize
                                               ])
        
        self.transform_te = transforms.Compose([
                                               lambda x: Image.fromarray(x).convert('RGB'),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               self.normalize
                                               ])

    def set_optimizer_parameters(self):
        # *** Optimizer
        self.adam = True
        self.learning_rate = 0.0001
        self.lr_decay_epochs = [30, 50, 100]
        self.lr_decay_rate = 0.1
        self.momentum = 0.9
        self.weight_decay = 0.0005

        # *** Scheduler
        self.scheduler = 'cosine'

    def set_network(self):
        self.model = 'T_v0'
        # self.backbone = 'vgg16bn'
        self.backbone = 'vitB16'
        self.ckpt = None

    def set_training_opts(self):
        # *** Print Option
        self.val_freq = 3
        self.print_freq = 50

        # *** Training
        self.batch_size = 32
        self.num_workers = 1
        self.epochs = 100

        # *** Save option
        self.save_freq = 100
        self.wandb = False

    def set_test_opts(self):
        self.ckpt = None
