import torch

from modules.base_utils.datasets import get_matching_datasets, get_n_classes, pick_poisoner
from modules.base_utils.util import load_model, needs_big_ims, clf_eval

user_model_flag = "r32p"
dataset_flag = "cifar"
clean_label = 9
target_label = 4
poisoner_flag = "1xs"


poisoner = pick_poisoner(poisoner_flag, dataset_flag, target_label)
big_ims = needs_big_ims(user_model_flag)
_, distillation, test, poison_test, _ =\
    get_matching_datasets(dataset_flag, poisoner, clean_label, big=big_ims)

model_location = "/Data/iuliia.korotkova/FLIP/experiments/example_downstream/model.pth"
model_retrain = load_model(user_model_flag, get_n_classes(dataset_flag))
model_retrain.load_state_dict(torch.load(model_location, map_location="cuda"))
model_retrain = model_retrain.cuda()

# Evaluate
print("Evaluating...")
clean_test_acc = clf_eval(model_retrain, test)[0]
poison_test_acc = clf_eval(model_retrain, poison_test.poison_dataset)[0]
print(f"{clean_test_acc=}")
print(f"{poison_test_acc=}")