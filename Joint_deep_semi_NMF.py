import pandas as pd
import tensorflow as tf
import numpy as np
import math

class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience  = patience
        self.verbose = verbose
 
    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('Training process is stopped early')
                return True
        else:
            self._step = 0
            self._loss = loss
 
        return False

def frob(z):
    vec = tf.reshape(z, [-1])
    return tf.reduce_sum(tf.multiply(vec, vec))		

# Load input data
add1 = pd.read_csv('../add1.csv', index_col=0)
add2 = pd.read_csv('../add2.csv', index_col=0)

me = pd.read_csv('../me.csv')
me = me.set_index('Unnamed: 0')


add1 = add1.ix[:,add1.columns.str.contains('AD|CTL')]
add2 = add2.ix[:,add2.columns.str.contains('AD|CTL')]

ge = pd.concat([add1, add2], axis=1)
ge = ge.ix[:,ge.columns.str.contains('AD|CTL')]


GE_data = np.array(ge).astype('float32')
DM_data = np.array(me).astype('float32')

# Hyperparameters
max_steps = 1000000
early_stopping = EarlyStopping(patience=200, verbose=1)

first_reduced_dimension = 100
second_reduced_dimension = 95
third_reduced_dimension = 90
fourth_reduced_dimension = 85

lambda_ = 0.001
       
n,dm = DM_data.shape
n,dg = GE_data.shape 

tf.set_random_seed(1)
sess = tf.InteractiveSession()

DM = tf.placeholder(tf.float32, shape=(None, dm))
GE = tf.placeholder(tf.float32, shape=(None, dg))

# Initialization using SVD
DM_svd_u_1, _, DM_svd_vh_1 = np.linalg.svd(DM_data,full_matrices=False)
DM_svd_u_2, _, DM_svd_vh_2 = np.linalg.svd(DM_svd_u_1,full_matrices=False)
DM_svd_u_3, _, DM_svd_vh_3 = np.linalg.svd(DM_svd_u_2,full_matrices=False)
DM_svd_u_4, _, DM_svd_vh_4 = np.linalg.svd(DM_svd_u_3,full_matrices=False)

GE_svd_u_1, _, GE_svd_vh_1 = np.linalg.svd(GE_data,full_matrices=False)
GE_svd_u_2, _, GE_svd_vh_2 = np.linalg.svd(GE_svd_u_1,full_matrices=False)
GE_svd_u_3, _, GE_svd_vh_3 = np.linalg.svd(GE_svd_u_2,full_matrices=False)
GE_svd_u_4, _, GE_svd_vh_4 = np.linalg.svd(GE_svd_u_3,full_matrices=False)
            

U = tf.Variable(tf.cast(DM_svd_u_4[:, 0:first_reduced_dimension],tf.float32))

Z21 = tf.Variable(tf.cast(DM_svd_u_2[0:first_reduced_dimension, 0:second_reduced_dimension],tf.float32))  
Z11 = tf.Variable(tf.cast(GE_svd_u_2[0:first_reduced_dimension, 0:second_reduced_dimension],tf.float32))

Z22 = tf.Variable(tf.cast(DM_svd_u_3[0:second_reduced_dimension, 0:third_reduced_dimension],tf.float32))
Z12 = tf.Variable(tf.cast(GE_svd_u_3[0:second_reduced_dimension, 0:third_reduced_dimension],tf.float32))

Z23 = tf.Variable(tf.cast(DM_svd_u_4[0:third_reduced_dimension, 0:third_reduced_dimension],tf.float32))  
Z13 = tf.Variable(tf.cast(GE_svd_u_4[0:third_reduced_dimension, 0:third_reduced_dimension],tf.float32))

H23 = tf.Variable(tf.cast(DM_svd_vh_1[0:fourth_reduced_dimension, :],tf.float32))
H13 = tf.Variable(tf.cast(GE_svd_vh_1[0:fourth_reduced_dimension, :],tf.float32))

# loss function
loss = frob(GE - tf.matmul(U, tf.sigmoid(tf.matmul(Z11, tf.sigmoid(tf.matmul(Z12, tf.sigmoid(tf.matmul(Z13, H13))))))))+\
	frob(DM - tf.matmul(U, tf.sigmoid(tf.matmul(Z21, tf.sigmoid(tf.matmul(Z22, tf.sigmoid(tf.matmul(Z23, H23))))))))+\
	lambda_*(frob(U) + frob(Z11) + frob(Z12) + frob(Z13) + frob(H13) + frob(Z21) + frob(Z22) + frob(Z23) + frob(H23))

diff_DM = frob(DM - tf.matmul(U, tf.sigmoid(tf.matmul(Z21, tf.sigmoid(tf.matmul(Z22, tf.sigmoid(tf.matmul(Z23, H23))))))))
diff_GE = frob(GE - tf.matmul(U, tf.sigmoid(tf.matmul(Z11, tf.sigmoid(tf.matmul(Z12, tf.sigmoid(tf.matmul(Z13, H13))))))))

MF = frob(GE - tf.matmul(U, tf.sigmoid(tf.matmul(Z11, tf.sigmoid(tf.matmul(Z12, tf.sigmoid(tf.matmul(Z13, H13))))))))+\
	frob(DM - tf.matmul(U, tf.sigmoid(tf.matmul(Z21, tf.sigmoid(tf.matmul(Z22, tf.sigmoid(tf.matmul(Z23, H23))))))))
L2 = lambda_*(frob(U) + frob(Z11) + frob(Z12) + frob(Z13) + frob(H13) + frob(Z21) + frob(Z22) + frob(Z23) + frob(H23))

train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

tf.global_variables_initializer().run()

funval = []
_, loss_iter = sess.run([train_step, loss],feed_dict={DM:DM_data,GE:GE_data })
funval.append(loss_iter)
for i in range(max_steps):
    _, loss_iter = sess.run([train_step, loss],feed_dict={DM:DM_data,GE:GE_data})
    funval.append(loss_iter)
    if(i%5000==0):
        print (i, " loss : %f" %sess.run(loss,feed_dict={DM:DM_data,GE:GE_data}))
        print (" Average diff_me : %f" %sess.run(diff_DM,feed_dict={DM:DM_data,GE:GE_data})) 
        print (" Average diff_ge : %f" %sess.run(diff_GE,feed_dict={DM:DM_data,GE:GE_data})) 
        print (" MF: %f  l2: %f" %(sess.run(MF, feed_dict={DM:DM_data,GE:GE_data}),sess.run(L2, feed_dict={DM:DM_data,GE:GE_data})))
    if early_stopping.validate(loss_iter):
        break
    if math.isnan(loss_iter):
        break



H20 = sess.run(tf.sigmoid(tf.matmul(Z21, tf.sigmoid(tf.matmul(Z22, tf.sigmoid(tf.matmul(Z23, H23)))))),feed_dict={DM:DM_data,GE:GE_data})
H10 = sess.run(tf.sigmoid(tf.matmul(Z11, tf.sigmoid(tf.matmul(Z12, tf.sigmoid(tf.matmul(Z13, H13)))))),feed_dict={DM:DM_data,GE:GE_data})
U_ = sess.run(U,feed_dict={DM:DM_data,GE:GE_data})

np.savetxt("../me_latent.txt", H20, delimiter="\t")
np.savetxt("../ge_latent.txt", H10, delimiter="\t")
np.savetxt("../samples_latent.txt", U_, delimiter="\t")

tf.reset_default_graph()
sess.close()

