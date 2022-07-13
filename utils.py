import math
import torch
from model import FnnLm, RNN, MyRnn, Atten_RNN, Lstm, Atten_Lstm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 构建词表
def make_dict(train_path):
    text = open(train_path, 'r', encoding='utf-8')  # open the train file
    word_list = set()  # a set for making dict

    for line in text:
        line = line.strip().split(" ")  # 去掉空白符并用
        word_list = word_list.union(set(line))

    word_list = list(sorted(word_list))  # set to list

    word2number_dict = {w: i + 2 for i, w in enumerate(word_list)}  # 以序号为索引的表
    number2word_dict = {i + 2: w for i, w in enumerate(word_list)}  # 以单词为索引的表

    # add the <pad> and <unk_word>
    word2number_dict["<pad>"] = 0
    number2word_dict[0] = "<pad>"
    word2number_dict["<unk_word>"] = 1
    number2word_dict[1] = "<unk_word>"

    return word2number_dict, number2word_dict


# 构建数据对
def make_batch(train_path, word2number_dict, batch_size, n_step):
    def word2number(n):
        try:
            return word2number_dict[n]
        except:
            return 1  # <unk_word>

    all_input_batch = []
    all_target_batch = []

    text = open(train_path, 'r', encoding='utf-8')  # open the file

    input_batch = []
    target_batch = []
    for sen in text:
        word = sen.strip().split(" ")  # space tokenizer

        if len(word) <= n_step:  # pad the sentence
            word = ["<pad>"] * (n_step + 1 - len(word)) + word

        for word_index in range(len(word) - n_step):
            input = [word2number(n) for n in word[word_index:word_index + n_step]]  # create (1~n-1) as input
            target = word2number(word[word_index + n_step])
            # create (n) as target, We usually call this 'casual language model'
            input_batch.append(input)
            target_batch.append(target)

            if len(input_batch) == batch_size:
                all_input_batch.append(input_batch)
                all_target_batch.append(target_batch)
                input_batch = []
                target_batch = []

    return all_input_batch, all_target_batch


# 获取数据
def get_data(data_path, word2number_dict, n_step, batch_size):
    all_input_batch, all_target_batch = make_batch(data_path, word2number_dict, batch_size, n_step)

    all_input_batch = torch.LongTensor(all_input_batch).to(device)  # list to tensor
    all_target_batch = torch.LongTensor(all_target_batch).to(device)

    return [all_input_batch, all_target_batch]


# 训练
def train_model(train_set, valid_set, model, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    # train
    for e in range(1, 201):
        all_input_train = train_set[0]
        all_target_train = train_set[1]
        total_train = len(list(all_input_train))
        count_train = 0
        model.train()
        total_loss = 0
        for input_batch, target_batch in zip(all_input_train, all_target_train):
            optimizer.zero_grad()
            output = model(input_batch)
            loss = criterion(output, target_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            count_train += 1
        print("epoch:{},train loss={:.6f}".format(e,total_loss/count_train))
        all_input_valid = valid_set[0]
        all_target_valid = valid_set[1]
        model.eval()
        count_valid = 0
        with torch.no_grad():
            total_loss = 0
            for input_batch, target_batch in zip(all_input_valid, all_target_valid):
                output = model(input_batch)
                loss = criterion(output, target_batch)
                total_loss += loss.item()
                count_valid += 1
            print("loss in valid set:{:.6f}".format(total_loss / count_valid))
            print("ppl in valid set:{:.6f}".format(math.exp(total_loss / count_valid)))

            if e % 10 == 0:
                torch.save(model, f'./model/nnlm_epoch{e}.ckpt')
                print('=>save model!!!')


def test_model(test_set, model):
    criterion = torch.nn.CrossEntropyLoss()
    all_input_test = test_set[0]
    all_target_test = test_set[1]
    model.eval()
    with torch.no_grad():
        total_loss = 0
        count_test = 1
        for (test_batch, test_target) in zip(all_input_test, all_target_test):
            output = model(test_batch)
            loss = criterion(output, test_target)
            total_loss += loss.item()
            count_test += 1
        print("ppl={:.6f}".format(math.exp(total_loss / count_test)))


# 获取模型
def get_model(args, n_class):
    if args.model == "fnn":
        model = FnnLm(n_class, args.emb_size, args.step, args.hidden).to(device)
        return model
    elif args.model == "rnn":
        model = RNN(n_class, args.emb_size, args.hidden).to(device)
        return model
    elif args.model == "atten_rnn":
        model = Atten_RNN(n_class, args.emb_size, args.hidden).to(device)
        return model
    elif args.model == "lstm":
        model = Lstm(n_class, args.emb_size, args.hidden).to(device)
        return model
    elif args.model == "atten_lstm":
        model = Atten_Lstm(n_class, args.emb_size, args.hidden).to(device)
        return model
    elif args.model == "myrnn":
        model = MyRnn(n_class, args.emb_size, args.hidden).to(device)
        return model
