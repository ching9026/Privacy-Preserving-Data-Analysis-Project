import numpy as np
from tqdm import tqdm
from torch import optim, nn

from trainers.fedbase import BaseFedarated
from models.ST_Net import STNET
from clients.base_client import BaseClient
from clients.target_client_es import TargetClientES
from utils.data_utils import traffic_dataset
from utils.data_utils_es import getTargetTrainValTestDataSet_ES


class pFedCTP_ES(BaseFedarated):
    """
    pFedCTP + Target-city Early Stopping.

    不修改作者原本的 trainers/pFedCTP.py。
    """

    def __init__(self, options):
        super(pFedCTP_ES, self).__init__(options=options)

    def setup_clients(self):
        all_clients = []
        client_id = 0

        print("target city: " + self.options["target_city"])

        for d in self.options["data_list"]:
            model = STNET(self.options, d).to(self.device)

            if d == self.options["target_city"]:
                train_dataset, val_dataset, test_dataset = getTargetTrainValTestDataSet_ES(
                    self.options,
                    d,
                )

                c = TargetClientES(
                    id=0,
                    name=d,
                    options=self.options,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    test_dataset=test_dataset,
                    model=model,
                )

            else:
                traffic_data = traffic_dataset(self.options, d)

                c = BaseClient(
                    id=client_id,
                    name=d,
                    options=self.options,
                    dataset=traffic_data,
                    model=model,
                )

            client_id += 1
            all_clients.append(c)

        return all_clients

    def trainClients(self):
        client_loss = []

        for c in self.train_clients:
            if self.options["SharePart"]:
                c.set_ada_shared_parameters_list(self.latest_model)
            else:
                c.adaptive_set_parameters_list(self.latest_model)

            client_loss.append(c.train_client_sampleData())

        print(client_loss)

    def train(self):
        final_res = None

        for round_i in tqdm(range(self.num_rounds)):
            self.select_clients()
            self.trainClients()
            self.aggregate()

            print("---------------------------------")
            print(f"[Federated Round {round_i + 1}/{self.num_rounds}]")

            for c in self.train_clients:
                if c.name == self.options["target_city"]:
                    res = c.target_dataloader()
                    print("[Target Test Metric]")
                    print(res)
                    final_res = res

        return final_res