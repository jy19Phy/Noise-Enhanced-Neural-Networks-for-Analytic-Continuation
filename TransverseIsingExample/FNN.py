# /usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# trainig dataset 
Iput_G = np.genfromtxt("./DataBeta2/InputS_Beta2Train.csv",delimiter=',')
Iput_A = np.genfromtxt("./DataBeta2/OutputA_Beta2Train.csv",delimiter=',')
# traing eta = 0.10
noises = np.random.normal(loc=0.0,scale=0.010,size=(len(Iput_G),Iput_G.shape[1]))
Iput_G = np.copy(Iput_G *(1+noises) )

# test dataset 
Iput_GTest = np.genfromtxt("./DataBeta2/InputS_Beta2Test.csv",delimiter=',')
Iput_ATest = np.genfromtxt("./DataBeta2/OutputA_Beta2Test.csv",delimiter=',')
# test eta = 0.10
noises = np.random.normal(loc=0.0,scale=0.010,size=(len(Iput_GTest),Iput_GTest.shape[1]))
Iput_GTest =  np.copy( Iput_GTest*(1+noises) ) 


n_G = Iput_G.shape[1]
n_A = Iput_A.shape[1] 
print("Train sample length :" ,Iput_G.shape[0])
print("Test sample length :" ,Iput_GTest.shape[0])


# FNN
n_Decoder = np.array((n_G,int(n_A),int(n_A),int(n_A),int(n_A),int(n_A),n_A ))
print('Decoder Layer:'+str(n_Decoder))

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
Deh1  = tf.nn.relu(tf.matmul(  TrainG, DeW1 )+ Deb1 )
Deh2  = tf.nn.relu(tf.matmul(  Deh1 , DeW2 )+ Deb2 )
Deh3  = tf.nn.relu(tf.matmul(  Deh2 , DeW3 )+ Deb3 )
Deh4  = tf.nn.relu(tf.matmul(  Deh3 , DeW4 )+ Deb4 )
Deh5  = tf.nn.relu(tf.matmul(  Deh4 , DeW5 )+ Deb5 )
ANN = tf.matmul(  Deh5 , DeW6 )+ Deb6
DeANN = tf.square(ANN) 


# training cost: learing rate will be adjusted manually during the training process 
DeAcost = tf.reduce_mean( tf.square(TrainA -  DeANN) )
training_opDecoder = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(DeAcost)
# loss
DeAloss = tf.reduce_mean(tf.abs(TrainA - DeANN))
DeAlossS = tf.reduce_mean(tf.abs(TrainA - DeANN),axis=1)



saver=tf.train.Saver()
with tf.Session()  as sess:
	ini=tf.global_variables_initializer()
	epoch = 0

	if epoch==0:
		ini.run()
		np.savetxt('./ResNN/Iput_GNoise.csv',Iput_G,delimiter=',')
		np.savetxt('./ResNN/Iput_GTest.csv',Iput_GTest,delimiter=',')
		np.savetxt('./ResNN/Iput_ATest.csv',Iput_ATest, delimiter=',')
	else:
		Iput_G     = np.genfromtxt('./ResNN/Iput_GNoise.csv',delimiter=',')
		Iput_GTest = np.genfromtxt('./ResNN/Iput_GTest.csv', delimiter=',')
		Iput_ATest = np.genfromtxt('./ResNN/Iput_ATest.csv', delimiter=',')
		saver.restore(sess, "./ResNN/my_model"+str(epoch)+".ckpt")

	batch_size=1000
	Train_X = Iput_G
	Train_Y = Iput_A
	Train_XTest = Iput_GTest
	Train_YTest = Iput_ATest

	loss_val=1.0
	while loss_val> 1e-5 and epoch < 200000:
		epoch=epoch+1
		shuffled_indices=np.random.permutation(len(Train_X))
		for step in range(0,len(Train_X)-batch_size,batch_size):
			batchX = Train_X[shuffled_indices[step:step+batch_size]]
			batchY = Train_Y[shuffled_indices[step:step+batch_size]]
			sess.run(training_opDecoder, feed_dict= {TrainG: batchX, TrainA: batchY})
		loss_DeA     = sess.run(DeAloss, feed_dict= {TrainG: Train_X, TrainA: Train_Y})
		Testloss_DeA = sess.run(DeAloss, feed_dict= {TrainG: Train_XTest, TrainA: Train_YTest})
		print('Epoch= ',epoch, '\tDeA_loss= ', loss_DeA,'\tTest_DeA_loss= ', Testloss_DeA)
		with open("./ResNN/loss.dat", "a+",  buffering=1000000) as file:
				file.write('\n'+str(epoch)+'\t'+str(loss_DeA)+'\t'+str(Testloss_DeA))
		if epoch%500==0:
			save_path=saver.save(sess, "./ResNN/my_model"+str(epoch)+".ckpt")
	save_path=saver.save(sess, "./ResNN/my_model"+str(epoch)+".ckpt")

	
	YNN 	=	sess.run(DeANN, feed_dict={TrainG: Iput_G})
	YNNErr 	= 	sess.run(DeAlossS, 	feed_dict={TrainG: Iput_G, TrainA: Iput_A})
	np.savetxt('./ResNN/DeANN.csv',YNN,delimiter=',')
	np.savetxt('./ResNN/DeANNErr.csv',YNNErr,delimiter=',')

	YNN 	=	sess.run(DeANN, feed_dict={TrainG: Iput_GTest}) 
	YNNErr 	= 	sess.run(DeAlossS, 	feed_dict={TrainG: Iput_GTest, TrainA: Iput_ATest})
	np.savetxt('./ResNN/DeANNTest.csv',YNN,delimiter=',')
	np.savetxt('./ResNN/DeANNErrTest.csv',YNNErr,delimiter=',')
	
	
	

