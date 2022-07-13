import torch
import argparse
from utils import get_data, get_model, make_dict, train_model, test_model

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 初始化超参
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train", help="train or test")
    parser.add_argument('--step', type=int, default=4, help="length of slip window")
    parser.add_argument('--hidden', type=int, default=8, help="number of hidden units")
    parser.add_argument('--emb_size', type=int, default=8, help="embedding size")
    parser.add_argument('--batch_size', type=int, default=512, help="batch_size")
    parser.add_argument('--lr', type=float, default=5e-4, help="learning rate")
    parser.add_argument('--train_path', type=str, default="./penn/train.txt", help="source of train data")
    parser.add_argument('--valid_path', type=str, default="./penn/valid.txt", help="source of valid data")
    parser.add_argument("--model", type=str, choices=["fnn", "rnn", "myrnn", "atten_rnn", "lstm", "atten_lstm"],
                        default="rnn",
                        help="choose model")
    parser.add_argument('--test_path', type=str, default="./penn/test.txt", help="source of valid data")
    parser.add_argument('--ckpt', type=str, default='./model/nnlm_epoch200.ckpt', help="model path")
    args = parser.parse_args()
    word2number_dict, number2word_dict = make_dict(args.train_path)
    print("dictionary size=> train:{}".format(len(word2number_dict)))
    n_class = len(word2number_dict)
    if args.mode == "train":
        print("\n====>Train")
        train_set = get_data(args.train_path, word2number_dict, args.step, args.batch_size)
        valid_set = get_data(args.valid_path, word2number_dict, args.step, 512)
        loss_func = torch.nn.CrossEntropyLoss()
        model = get_model(args, n_class)
        train_model(train_set, valid_set, model, args.lr)
    elif args.mode == "test":
        test_set = get_data(args.test_path, word2number_dict, args.step, 512)
        print("\n===>Test")
        model = torch.load(args.ckpt, map_location=device)
        test_model(test_set, model)
