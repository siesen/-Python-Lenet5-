import numpy as np
from lenet5 import Lenet5
import matplotlib.pyplot as plt
import tensorflow as tf
from trainer import Trainer

#载入数据集
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
#增加维度
x_train=np.expand_dims(x_train,axis=1)
x_test=np.expand_dims(x_test,axis=1)
x_train=x_train/255.0
x_test=x_test/255.0

#独热编码
temp=np.zeros((y_train.size,10))
temp[np.arange(y_train.size),y_train]=1
y_train=temp
temp=np.zeros((y_test.size,10))
temp[np.arange(y_test.size),y_test]=1
y_test=temp

      
max_epochs = 20

network = Lenet5(input_dim=(1,28,28), 
                 conv_param={'filter_num1':6, 'filter_size1':3,'filter_num2':16, 'filter_size2':3, 'pad':1, 'stride':1},
                 hidden_size1=120,hidden_size2=84, output_size=10, weight_init_std=0.01)

                        
trainer = Trainer(network, x_train, y_train, x_test, y_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("params.pkl")
print("Saved Network Parameters!")

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()



