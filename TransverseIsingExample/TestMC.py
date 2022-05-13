# /usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


Iput_GTestMC = np.genfromtxt("./DataBeta2/MC/G_MC.csv",delimiter=',')
Iput_ATestMC = np.genfromtxt("./DataBeta2/MC/A_MC.csv",delimiter=',')


n_G = Iput_GTestMC.shape[1]
n_A = Iput_ATestMC.shape[1] 


n_Decoder = np.array((n_G,int(n_A),int(n_A),int(n_A),int(n_A),int(n_A),n_A ))
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

# training cost 
DeAcost = tf.reduce_mean( tf.square(TrainA -  DeANN) )
training_opDecoder = tf.train.AdamOptimizer(learning_rate=0.001).minimize(DeAcost)
# loss
DeAloss = tf.reduce_mean(tf.abs(TrainA - DeANN))
DeAlossS = tf.reduce_mean(tf.abs(TrainA - DeANN),axis=1)




saver=tf.train.Saver()
with tf.Session()  as sess:

	# provide the finally epoch which save the training result of the model 
	epoch = 300000
	# provide the corresponding training eta of the trained model 
	etaTrain = 0.010
	# Import the trained model 
	saver.restore(sess, "./ResNN6/my_model"+str(epoch)+".ckpt")


	TestlossMC = sess.run(DeAloss, feed_dict= {TrainG: Iput_GTestMC, TrainA: Iput_ATestMC})
	with open("./ResNN6/TestMC/lossTestMC.dat","a+",  buffering=1000000) as file:
				file.write(str(etaTrain)+'\t'+str(TestlossMC))

	