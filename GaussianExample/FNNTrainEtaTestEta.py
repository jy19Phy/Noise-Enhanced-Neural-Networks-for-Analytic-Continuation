# /usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# input training dataset 
wList  = np.genfromtxt("./Data/OmegaList.csv",delimiter=',')
wList= np.reshape(wList, (1,-1))
Iput_G = np.genfromtxt("./Data/DataG100000.csv",delimiter=',')
Iput_A = np.genfromtxt("./Data/DataA100000.csv",delimiter=',')
Iput_A = Iput_A/ np.sum(Iput_A,axis=1,keepdims=True)

# train eta 0.001 (eta_train = 0.01 in Fig 2a)
noises = np.random.normal(loc=0.0,scale=0.01,size=(len(Iput_G),Iput_G.shape[1]))
Iput_G =np.copy( Iput_G*(1+noises) )
np.savetxt('./ResNN/Iput_G.csv',Iput_G,delimiter=',')

# input tests dataset
Iput_GTest = np.genfromtxt("./Data/DataGN3S5000Test.csv",delimiter=',')
Iput_ATest = np.genfromtxt("./Data/DataAN3S5000Test.csv",delimiter=',')
Iput_ATest = Iput_ATest/ np.sum(Iput_ATest,axis=1,keepdims=True)
# test eat 0.001 (eta_test = 0.01 in Fig 2a)
noises = np.random.normal(loc=0.0,scale=0.01,size=(len(Iput_GTest),Iput_GTest.shape[1]))
Iput_GTest =  np.copy( Iput_GTest*(1+noises) ) 
np.savetxt('./ResNN/Iput_GTest.csv',Iput_GTest,delimiter=',')

n_G = Iput_G.shape[1]
n_A = Iput_A.shape[1] 

# FNN (Fig 1)
n_Decoder = np.array((n_G,n_A,n_A,n_A,n_A,n_A, n_A ))
TrainA = tf.placeholder( tf.float32, [None,n_A])
TrainG = tf.placeholder( tf.float32, [None,n_G])
DeW1  = tf.Variable(tf.truncated_normal([n_Decoder[0], n_Decoder[1] ], stddev=0.1 ))
Deb1  = tf.Variable(tf.truncated_normal(      		  [n_Decoder[1] ], stddev=0.001 ))
DeW2  = tf.Variable(tf.truncated_normal([n_Decoder[1] ,n_Decoder[2] ], stddev=0.1 ))
Deb2  = tf.Variable(tf.truncated_normal(      		  [n_Decoder[2] ], stddev=0.001 ))
DeW3  = tf.Variable(tf.truncated_normal([n_Decoder[2] ,n_Decoder[3] ], stddev=0.1 ))
Deb3  = tf.Variable(tf.truncated_normal(      		  [n_Decoder[3] ], stddev=0.001 ))
DeW4  = tf.Variable(tf.truncated_normal([n_Decoder[3] ,n_Decoder[4] ], stddev=0.1 ))
Deb4  = tf.Variable(tf.truncated_normal(      		  [n_Decoder[4] ], stddev=0.001 ))
DeW5  = tf.Variable(tf.truncated_normal([n_Decoder[4] ,n_Decoder[5] ], stddev=0.1 ))
Deb5  = tf.Variable(tf.truncated_normal(      		  [n_Decoder[5] ], stddev=0.001 ))
DeW6  = tf.Variable(tf.truncated_normal([n_Decoder[5] ,n_Decoder[6] ], stddev=0.1 ))
Deb6  = tf.Variable(tf.truncated_normal(      		  [n_Decoder[6] ], stddev=0.001 ))
Deh1  = tf.nn.relu(tf.matmul(  TrainG  , DeW1 )+ Deb1 )
Deh2  = tf.nn.relu(tf.matmul(  Deh1 , DeW2 )+ Deb2 )
Deh3  = tf.nn.relu(tf.matmul(  Deh2 , DeW3 )+ Deb3 )
Deh4  = tf.nn.relu(tf.matmul(  Deh3 , DeW4 )+ Deb4 )
Deh5  = tf.nn.relu(tf.matmul(  Deh4 , DeW5 )+ Deb5 )
DeANN = tf.nn.softmax(tf.matmul(  Deh5 , DeW6 )+ Deb6 )

# training cost: KLD (Eq. 4 )
DeAcost = tf.reduce_mean( TrainA*tf.log( (tf.clip_by_value( TrainA, 1e-15, 1e10)) ) - TrainA* tf.log( (tf.clip_by_value( DeANN, 1e-15, 1e10)) ) )
training_opDecoder = tf.train.AdamOptimizer(learning_rate=0.001).minimize(DeAcost)
# loss function: MAV (Eq. 5)
DeAloss = tf.reduce_mean(tf.abs(DeANN-TrainA))
DeAlossS = tf.reduce_mean(tf.abs(DeANN-TrainA),axis=1)


saver=tf.train.Saver()
with tf.Session()  as sess:
	ini=tf.global_variables_initializer()
	epoch = 0

	if epoch==0:
		ini.run()
	else:
		saver.restore(sess, "./ResNN/my_model"+str(epoch)+".ckpt")

	batch_size=10000
	Train_X = Iput_A
	Train_Y = Iput_G
	Train_XTest = Iput_ATest
	Train_YTest = Iput_GTest

	loss_val=1.0
	while loss_val> 1e-5 and epoch < 10000:
		epoch=epoch+1
		shuffled_indices=np.random.permutation(len(Train_X))
		for step in range(0,len(Train_X)-batch_size,batch_size):
			batchX = Train_X[shuffled_indices[step:step+batch_size]]
			batchY = Train_Y[shuffled_indices[step:step+batch_size]]
			sess.run(training_opDecoder,  	feed_dict= {TrainA: batchX, TrainG: batchY})
		loss_DeA = sess.run(DeAloss, feed_dict= {TrainG: Train_Y, TrainA: Train_X})
		Testloss_DeA = sess.run(DeAloss, feed_dict= {TrainG: Train_YTest, TrainA: Train_XTest})
		print('Epoch= ',epoch, '\tDeA_loss= ', loss_DeA,'\tTest_DeA_loss= ', Testloss_DeA)
		with open("./ResNN/loss.dat", "a+",  buffering=1000000) as file:
				file.write('\n'+str(epoch)+'\t'+str(loss_DeA)+'\t'+str(Testloss_DeA))
		if epoch%500==0:
			save_path=saver.save(sess, "./ResNN/my_model"+str(epoch)+".ckpt")
	save_path=saver.save(sess, "./ResNN/my_model"+str(epoch)+".ckpt")

# Training dataset
	YNN 	=	sess.run(DeANN, feed_dict={TrainG: Iput_G})
	np.savetxt('./ResNN/ANNTrain.csv',YNN,delimiter=',')
	YNNErr 	= 	sess.run(DeAlossS, 	feed_dict={TrainG: Iput_G, TrainA: Iput_A})
	np.savetxt('./ResNN/ANNTrainErr.csv',YNNErr,delimiter=',')

# Test dataset 
	YNN 	=	sess.run(DeANN, feed_dict={TrainG: Iput_GTest})
	YNNErr 	= 	sess.run(DeAlossS, 	feed_dict={TrainG: Iput_GTest, TrainA: Iput_ATest})
	np.savetxt('./ResNN/DeANNTest.csv',YNN,delimiter=',')
	np.savetxt('./ResNN/DeANNTestErr.csv',YNNErr,delimiter=',')

	

	

	

