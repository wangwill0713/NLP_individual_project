import matplotlib.pyplot as plt

# 实验一:调整滑动窗口大小
n_step = [2, 4, 6, 8, 10]
fnn_small_valid = [639.955845, 780.808157, 825.335049, 946.991538, 985.524195]
fnn_small_test = [511.325236, 626.161278, 701.007826, 764.720403, 750.458595]
fnn_ori_valid = [388.548629, 395.729012, 389.150235, 427.13198, 458.745119]
fnn_ori_test = [357.200508, 357.54884, 355.204893, 386.888923, 414.954034]
rnn_small_valid = [770.308629, 758.102631, 734.719048, 739.81947, 779.654232]
rnn_small_test = [608.997995, 598.734032, 580.651017, 568.09187, 592.041854]
rnn_ori_valid = [563.210666, 426.901313, 422.523487, 424.264386, 428.849289]
rnn_ori_test = [521.644523, 395.822396, 391.336019, 390.620832, 399.519352]
plt.figure(1, (8, 6))
plt.subplot(1, 2, 1)
plt.title('ppl in valid_set')
plt.plot(n_step, fnn_small_valid, '.--', label='FNN in small')
plt.plot(n_step, rnn_small_valid, '*--', label='RNN in small')
plt.plot(n_step, fnn_ori_valid, '.-', label='FNN in origin')
plt.plot(n_step, rnn_ori_valid, '*-', label='RNN in origin')
plt.xticks(n_step)
plt.xlabel('window step')
plt.ylabel('ppl')
plt.legend()
plt.subplot(1, 2, 2)
plt.title('ppl in test_set')
plt.plot(n_step, fnn_small_test, '.--', label='FNN in small')
plt.plot(n_step, rnn_small_test, '*--', label='RNN in small')
plt.plot(n_step, fnn_ori_test, '.-', label='FNN in origin')
plt.plot(n_step, rnn_ori_test, '*-', label='RNN in origin')
plt.xticks(n_step)
plt.xlabel('window step')
plt.ylabel('ppl')
plt.legend()
plt.show()
# 实验二:学习率的调整
lr = [1e-3, 7.5e-4, 5e-4, 2.5e-4, 1e-4]
fnn_valid = [395.729012, 391.097569, 391.500515, 376.224449, 418.205793]
fnn_test = [357.54884, 351.541213, 358.802717, 347.500073, 387.602223]
rnn_valid = [426.901313, 427.565669, 430.580581, 466.667521, 583.081799]
rnn_test = [395.822396, 397.07792, 401.139894, 436.967451, 540.950639]
plt.figure(2, (8, 6))
plt.subplot(1, 2, 1)
plt.title('ppl in valid_set')
plt.plot(lr, fnn_valid, '*-', label='FNN')
plt.plot(lr, rnn_valid, '.-', label='RNN')
plt.xticks(lr)
plt.xlabel('learning rate')
plt.ylabel('ppl')
plt.legend()
plt.subplot(1, 2, 2)
plt.title('ppl in test_set')
plt.plot(lr, fnn_test, '*-', label='FNN')
plt.plot(lr, rnn_test, '.-', label='RNN')
plt.xticks(lr)
plt.xlabel('learning rate')
plt.ylabel('ppl')
plt.legend()
plt.show()
# 实验三:隐藏层维数与隐藏层数
hidden_size = [2, 4, 8, 16, 24]
rnn1_valid = [430.853739, 379.093607, 320.697332, 287.021063, 259.263362]
rnn1_test = [400.765389, 351.212702, 295.932878, 263.692569, 240.181953]
rnn2_valid = [432.82552, 354.169273, 311.92597, 268.127542, 259.59715]
rnn2_test = [403.718098, 329.380901, 289.003134, 246.836274, 238.022343]
plt.figure(3, (8, 6))
plt.subplot(1, 2, 1)
plt.title('ppl in valid_set')
plt.plot(hidden_size, rnn1_valid, '*-', label='one hidden layer')
plt.plot(hidden_size, rnn2_valid, '.-', label='two hidden layers')
plt.xticks(hidden_size)
plt.xlabel('hidden_layer_size')
plt.ylabel('ppl')
plt.legend()
plt.subplot(1, 2, 2)
plt.title('ppl in test_set')
plt.plot(hidden_size, rnn1_test, '*-', label='one hidden layer')
plt.plot(hidden_size, rnn2_test, '.-', label='two hidden layers')
plt.xticks(hidden_size)
plt.xlabel('hidden_layer_size')
plt.ylabel('ppl')
plt.legend()
plt.show()
# 实验六:emb_size
emb_size = [2, 4, 8, 16, 32, 64]
fnn_valid_emb = [331.848746, 285.774922, 264.805558, 269.629707, 281.312286, 344.666934]
fnn_test_emb = [295.764385, 250.580771, 227.965297, 235.4183, 248.645736, 296.824148]
rnn_valid_emb = [346.175104, 309.153224, 300.433116, 284.844882, 275.649765, 267.89704]
rnn_test_emb = [305.706536, 278.182311, 267.1995, 251.46367, 243.230762, 235.456215]
plt.figure(4, (8, 6))
plt.subplot(1, 2, 1)
plt.title('ppl in valid_set')
plt.plot(emb_size, fnn_valid_emb, '*-', label='FNN')
plt.plot(emb_size, rnn_valid_emb, '.-', label='RNN')
plt.xticks(emb_size)
plt.xlabel('embedding_size')
plt.ylabel('ppl')
plt.legend()
plt.subplot(1, 2, 2)
plt.title('ppl in test_set')
plt.plot(emb_size, fnn_test_emb, '*-', label='FNN')
plt.plot(emb_size, rnn_test_emb, '.-', label='RNN')
plt.xticks(emb_size)
plt.xlabel('embedding_size')
plt.ylabel('ppl')
plt.legend()
plt.show()
