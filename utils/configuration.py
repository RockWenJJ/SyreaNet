import logging
import os
import yaml
from datetime import datetime

class Config(object):
    def __init__(self, args):
        self.init_from_args(args)
        
    def init_from_args(self, args):
        f = open(args.config, 'r')
        self.cfgs = yaml.load(f, Loader=yaml.FullLoader)
        current_time = datetime.now().strftime("%m%d%H%M")
        if 'default' == self.cfgs["name"]:
            self.cfgs["name"] = self.cfgs["name"] + "-" + current_time
        self.cfgs["config"] = args.config
        
        self.cuda = self.cfgs['cuda']
        self.config_file = args.config
        self.project = self.cfgs['project']
        self.name = self.cfgs['name']
        self.dataset = self.cfgs['Dataset']
        self.network = self.cfgs['Network']
        # self.optimizer = self.cfgs['Optimizer']
        # self.loss = self.cfgs['Loss']
        # self.train = self.cfgs['Train']
        self.prepare = self.cfgs['Prepare']
        # self.logger = self.cfgs['Logger']
        