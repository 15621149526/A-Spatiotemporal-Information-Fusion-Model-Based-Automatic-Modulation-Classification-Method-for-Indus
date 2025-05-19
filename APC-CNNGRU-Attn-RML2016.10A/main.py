import os
os.environ["KERAS_BACKEND"] = "tensorflow"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pickle
import keras
from keras.callbacks import LearningRateScheduler

from keras.layers import Dense,GRU
from keras.models import Sequential
import mltools,dataset2016
import rmlmodels.CNN2 as mcl

import csv
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
for gpu in tf.config.list_physical_devices('GPU'):
    print(gpu)
tf.config.run_functions_eagerly(True)
(mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx) = dataset2016.load_data()
X_train=np.expand_dims(X_train,axis=3)
X_test=np.expand_dims(X_test,axis=3)
X_val=np.expand_dims(X_val,axis=3)
print("xshap",X_train.shape)
classes = mods




nb_epoch =10
batch_size = 128

model=mcl.CNN2()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model.summary()


filepath = 'E:\shiyan\CNN2\APC-CNNGRU-Attn-RML2016.10A\weights\weights.h5'
history = model.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=1,
    validation_data=(X_val,Y_val),
    callbacks = [
                keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,verbose=1,patience=5,min_lr=0.0000001),
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')

                ]

                    )

def predict(model):

    confusion_dir = 'predictresult/confusion_matrices'
    if os.path.exists(confusion_dir):
        shutil.rmtree(confusion_dir)
    os.makedirs(confusion_dir, exist_ok=True)

    model.load_weights(filepath)




    test_Y_hat = model.predict(X_test, batch_size=batch_size)
    confnorm,_,_ = mltools.calculate_confusion_matrix(Y_test,test_Y_hat,classes)
    mltools.plot_confusion_matrix(confnorm, labels=['8PSK','AM-DSB','AM-SSB','BPSK','CPFSK','GFSK','4-PAM','16-QAM','64-QAM','QPSK','WBFM'],save_filename='figure/sclstm-a_total_confusion')




    acc = {}
    acc_mod_snr = np.zeros( (len(classes),len(snrs)) )
    i = 0
    for snr in snrs:


        test_SNRs = [lbl[x][1] for x in test_idx]

        test_X_i=X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i=Y_test[np.where(np.array(test_SNRs) == snr)]


        test_Y_i_hat = model.predict(test_X_i)

        confnorm_i,cor,ncor = mltools.calculate_confusion_matrix(test_Y_i,test_Y_i_hat,classes)
        acc[snr] = 1.0 * cor / (cor + ncor)
        result = cor / (cor + ncor)
        with open('acc111.csv', 'a', newline='') as f0:
            write0 = csv.writer(f0)
            write0.writerow([result])
        print("7898789", confnorm_i)
        mltools.plot_confusion_matrix(confnorm_i, labels=['8PSK','AM-DSB','AM-SSB','BPSK','CPFSK','GFSK','4-PAM','16-QAM','64-QAM','QPSK','WBFM'], title="Confusion Matrix",save_filename="figure/Confusion(SNR=%d)(ACC=%2f).png" % (snr,100.0*acc[snr]))
        print("sdasdad",confnorm_i)
        filename = os.path.join(confusion_dir, f'confusion_snr_{snr}.csv')
        np.savetxt(filename, confnorm_i, delimiter=',', fmt='%2f')
        print(f"覆盖保存: {filename}")


        acc_mod_snr[:,i] = np.round(np.diag(confnorm_i)/np.sum(confnorm_i,axis=1),3)
        i = i +1




    dis_num=11
    for g in range(int(np.ceil(acc_mod_snr.shape[0]/dis_num))):
        assert (0 <= dis_num <= acc_mod_snr.shape[0])
        beg_index = g*dis_num
        end_index = np.min([(g+1)*dis_num,acc_mod_snr.shape[0]])

        plt.figure(figsize=(12, 10))
        plt.xlabel("Signal to Noise Ratio")
        plt.ylabel("Classification Accuracy")
        plt.title("Classification Accuracy for Each Mod")

        for i in range(beg_index,end_index):
            plt.plot(snrs, acc_mod_snr[i], label=classes[i])

            for x, y in zip(snrs, acc_mod_snr[i]):
                plt.text(x, y, y, ha='center', va='bottom', fontsize=8)

        plt.legend()
        plt.grid()
        plt.savefig('figure/acc_with_mod_{}.png'.format(g+1))
        print("Saving figure for g+1:", g + 1)

        plt.close()

    # fd = open('predictresult/acc_for_mod.dat', 'wb')
    # pickle.dump((acc_mod_snr), fd)
    # fd.close()
    #
    # np.savetxt('predictresult/acc_mod_snr.csv', acc_mod_snr, delimiter=',', fmt='%.3f')
    #
    # with open('predictresult/classes.txt', 'w') as f:
    #     for mod in classes:
    #         f.write(f"{mod}\n")
    #
    #
    # print("acc",acc)
    # fd = open('predictresult/acc.dat','wb')
    # pickle.dump( (acc) , fd )
    #
    # with open('predictresult/acc.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['SNR', 'Accuracy'])
    #     for snr in snrs:
    #         writer.writerow([snr, acc[snr]])



    plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title(" Classification Accuracy on RadioML 2016.10 Alpha")
    plt.tight_layout()
    plt.savefig('figure/each_acc.png')
predict(model)

