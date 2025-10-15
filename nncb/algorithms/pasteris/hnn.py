import numpy as np

from nncb.algorithms.pasteris.cbnn import CBNNModel
import nncb.algorithms.pasteris.hnn_utils as utils
from nncb.utils.leaf_list import NavigatingNetList, LeafList

DEFAULT_LEAF_LIST_TYPE = NavigatingNetList

class HNNModel(CBNNModel):
# TODO: try storing the bt_B etc. on this tree instead? Primarily the bt_B?
    def __init__(self, rho, binning_radius, T, leaf_list_type=DEFAULT_LEAF_LIST_TYPE):
        super().__init__(rho, binning_radius, T)
        self.H_sets = None
        # self.dataset = None
        self.trial_number = 0

    def train(self, dataset):

        self.set_dataset(dataset)

        all_training_info = []

        H_sets = self.H_sets
        trial_number = self.trial_number

        if self.bt_B is None:
            bt_B = self.create_bt_B()
        else:
            bt_B = self.bt_B
            
        leaf_list = self.leaf_list

        if self.eta is None:
            self.K = len(self.actions)
            self.eta = self.calculate_learning_rate()

        bt_Z = self.bt_Z
        tt_H_Z = self.tt_H_Z

        for i, data in enumerate(dataset.dataset):
            x = data[:-1]
            label = data[-1]
            H_sets, bt_B, bt_Z, tt_H_Z, leaf_list, action, loss = utils.hnn(self, H_sets, bt_B, bt_Z, tt_H_Z, leaf_list, x, label, trial_number)

            if action is not None:
                all_training_info.append([action, loss])

            trial_number += 1

        self.H_sets = H_sets
        self.bt_B = bt_B
        self.bt_Z = bt_Z
        self.tt_H_Z = tt_H_Z
        self.leaf_list = leaf_list
        self.trial_number = trial_number

        return np.array(all_training_info)

    def predict(self, x):

        if self.H_sets is None:
            raise RuntimeError("Model not trained, run model.train(dataset) first.")


    def save_model(self, filepath):
        filepath += "_hnn"
        return super().save_model(filepath)
    
    def load_model(self, filepath):
        filepath += "_hnn"
        return super().load_model(filepath)


