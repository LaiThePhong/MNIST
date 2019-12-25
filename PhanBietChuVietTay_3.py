# 1. Thêm các thư viện cần thiết
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.datasets import mnist

#Định nghĩa về hàm softmax
def softmax_stable(Z):
# Z = Z.reshape(Z.shape[0], -1)
    e_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True))# Tránh overflow
    A = e_Z / e_Z.sum(axis = 1, keepdims = True)
    return A

# Định nghĩa về hàm mất mát
def softmax_loss(X, y, W):# y là giá trị đầu ra thực one-hot (là list của chỉ số những giá trị có xác suất max), là matrix cỡ (Nx1), X matrix cỡ Nxd, W matrix cỡ dxC
    A = softmax_stable(X.dot(W)) # A là giá trị dự đoán, có cỡ (NxC)
    id0 = range(X.shape[0]) # indexes in axis 0, indexes in axis 1 are in y
    k=np.argmax(y,axis=1) # Lấy chỉ số từ những hàng (mỗi hàng có C node giá trị) có gía trị 1 của y
    #⬆️ id0 chạy từ 0--> N-1
    return -np.mean(np.log(A[id0, k]))# Hàm log trả list(keepdims=False). Lấy giá trị trung bình của hàm mất mát của các điểm data, trả kq là 1 số thực

# Định nghĩa về Đạo hàm hàm mất mát
def softmax_grad(X, y, W):# y là one-hot
    A = softmax_stable(X.dot(W)) # shape of (N, C)
    id0 = range(X.shape[0])
    k=np.argmax(y,axis=1)
    A[id0,k]-=1 #A-Y,shapeof(N,C)
    return X.T.dot(A)/X.shape[0]

# Định nghĩa hàm dự đoán. Từ các hàng (axis=1) matrix, chọn ra chỉ số của giá trị xác suất max trong hàng đó. 
def pred(W, X):# Trả list cỡ (Nx1)
    return np.argmax(X.dot(W), axis =1)#X cỡ (Nxd), W cỡ (dxC)

# Định nghĩa hàm huấn luyện 
def softmax_fit(X, y, W, lr = 0.01, nepoches = 10, tol = 1e-5, batch_size = 10):
    acc_plot=[0]
    ep_plot=[0]
    W_old = W.copy()
    ep = 0
    loss_hist = [softmax_loss(X, y, W)] # store history of loss
    N = X.shape[0]
    nbatches = int(np.ceil(float(N)/batch_size))# Làm cho số nbatches sang số nguyên
    while ep < nepoches:
        mix_ids = np.random.permutation(N) # mix data. Xuất ra array data (List) sắp xếp ngẫu nhiên trong (0, N) 
        for i in range(nbatches):
            # get the i-th batch
            batch_ids = mix_ids[batch_size*i:min(batch_size*(i+1), N)]
            X_batch, y_batch = X[batch_ids], y[batch_ids]
            W -= lr*softmax_grad(X_batch, y_batch, W) # update gradient descent
        loss_hist.append(softmax_loss(X, y, W))

        ep_plot.append(ep)# Tạo list các epoch (để vẽ đồ thị)
        acc_plot.append(acc(W,X_bar_test,Y_test))# Tạo list các giá trị Accuracy (để vẽ đồ thị)
        ep += 1
        if np.linalg.norm(W - W_old)/W.size < tol:
            break
        #print('Số epoches:%d, Giá trị hàm loss:%f' %(ep, loss_hist[-1]))
        W_old = W.copy()
    return W, loss_hist,ep_plot,acc_plot

# Hàm tính độ chính xác
def acc (W,X,y):
    y_pred=pred(W, X) # Shape of (N,1)
    y=np.argmax(y,axis=1) # Đưa về one-hot shape of (N,1)
    acc=100*np.mean(y_pred==y)
    return acc

# 2. Load dữ liệu MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data() 
X_val, y_val = X_train[50000:60000,:], y_train[50000:60000] #shape of (10000,28,28)
X_train, y_train = X_train[:50000,:], y_train[:50000] # shape of (50000,28,28)

# Reshape lại dữ liệu. Giảm bớt giá trị dữ liệu (Vì màu xám có giá trị 0 or 255)
X_train=X_train.reshape(-1,28*28)/255
X_val=X_val.reshape(-1,28*28)/255
X_test=X_test.reshape(-1,28*28)/255

# Tạo input matrix mở rộng (Thêm phần tử 1 vào mỗi dữ liệu)
X_bar_train = np.concatenate((np.ones((X_train.shape[0],1)),X_train),axis=1) # shape of (50000, 785)
X_bar_val = np.concatenate((np.ones((X_val.shape[0],1)),X_val),axis=1) # shape of (50000, 785)
X_bar_test = np.concatenate((np.ones((X_test.shape[0],1)),X_test),axis=1) # shape of (50000, 785)

# 3. One hot encoding label (Y)
Y_train = np_utils.to_categorical(y_train, 10) # shape of (50000,10)
Y_val = np_utils.to_categorical(y_val, 10) # shape of (10000,10)
Y_test = np_utils.to_categorical(y_test, 10) # shape of (10000,10)

#4. Training data
# Khởi tạo matrix trọng số ban đầu ngẫu nhiên cỡ (785x10)
def W(X,y):# y là giá trị thực
     W = np.random.rand(X.shape[1],y.shape[1])
     return W

#Chạy trên training set

W_0_train=W(X_bar_train,Y_train)# Matrix trọng số ban đầu
#W_init=np.zeros((785,10))
W_train,loss_hist,ep_plot,acc_plot = softmax_fit(X_bar_train,Y_train,W_0_train,lr=0.1,nepoches=20,tol=1e-5,batch_size=10)

plt.xlim(xmax=12)
plt.ylim(ymax=100)
plt.plot(ep_plot,acc_plot)
plt.xlabel('epoch')
plt.ylabel('accuracy(%)')
#plt.legend()
plt.show()

print('Training accracy: %.2f ' % acc(W_train,X_bar_test,Y_test))
sys.exit(0)