import os
import argparse
from cell import utils

parser = argparse.ArgumentParser()
parser.add_argument("--N", default=0, type=int, help="number of walks per node")
parser.add_argument("--length", default=0, type=int, help="length of each walk")
parser.add_argument("--p", default=1, type=float, help="p")
parser.add_argument("--q", default=1, type=float, help="q")
parser.add_argument("--walk_filename", default=None, type=str, help="the file name to be used for the input")
parser.add_argument("--roi", default="VISp", type=str, help="region of interest")
parser.add_argument("--project_name", default=None, type=str, help="name of the project")
parser.add_argument("--layer_class", default=None, type=str, help="layer class, e.g single_layer")
parser.add_argument("--layer", default=None, type=str, help="layer name e.g base_unnormalized_allcombined")
parser.add_argument("--walk_type", default=None, type=str, help="e.g Directed_Weighted_node2vec")
parser.add_argument("--window", default=None, type=int, help="window size for contex-tuple pair")
parser.add_argument("--batch_size", default=None, type=int, help="batch size")
parser.add_argument("--embedding_size", default=None, type=int, help="embedding_size")
parser.add_argument("--learning_rate", default=None, type=float, help="learning_rate")
parser.add_argument("--n_epochs", default=1, type=int, help="n_epochs")
parser.add_argument("--opt_add", default=None, type=str, help="an additive name for loss and model file name")


def main(N, length, p, q, walk_filename, roi, project_name, layer_class, layer, walk_type, window, batch_size,
         embedding_size, learning_rate, n_epochs, opt_add):

    walk_dir = utils.get_walk_dir(roi, project_name, N, length, p, q, layer_class, layer, walk_type)
    walk_path = os.path.join(walk_dir, walk_filename)

    if not os.path.isfile(walk_path):
        raise ValueError("No such a walk file:", walk_path)
    else:
        print("Walk file exist and is being used")
        print("")
        print(walk_path)
        print("_________________________________")

    model_dir = utils.get_model_dir(project_name, roi, N, length, p, q, layer_class, layer, walk_type)

    if not os.path.isdir(model_dir):
        raise ValueError("No such a model dir exist:", model_dir)
    else:
        print("model dir exist and output will be written in this dir:")
        print("")
        print(model_dir)
        print("_______________________________________________________")

    model_name = utils.get_model_name(embedding_size, n_epochs, window, learning_rate, batch_size, opt_add)
    model_path = os.path.join(model_dir, model_name)

    if os.path.isfile(model_path):
        raise ValueError("model output already exist:", model_path)
    else:
        print("model output will be written in the following path:")
        print("")
        print(model_path)
        print("_______________________________________________________")

    loss_name = utils.get_loss_filename(embedding_size, n_epochs, window, learning_rate, batch_size, opt_add)
    loss_path = os.path.join(model_dir, loss_name)

    if os.path.isfile(loss_path):
        raise ValueError("loss output already exist:", loss_path)
    else:
        print("loss output will be written in the following path:")
        print("")
        print(loss_path)
        print("_______________________________________________________")

    pwd = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(pwd, "IO_path.csv")
    utils.write_list_to_csv(path, [walk_path, model_path, loss_path])


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
