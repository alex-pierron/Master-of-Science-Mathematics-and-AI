import streamlit as st
from MaskedConv import *
from PixelCNN_MNIST_model import PixelCNN_MNIST

#This files is the one to run to launch properly the streamlit application

header = st.container()

description = st.container()

selecting_model = st.container()


with header:
    st.title("PixelCNN application")
    st.markdown('''This application is here to illustrate our implementation of PixelCNN, a deep learning algorithm created for image generation an image completion. \n
For more detail about the algorithms and the papers related, you can read our full report on the topic''')
    
with selecting_model:

    st.header("We will train a model for MNIST")
    st.markdown('''You can choose several parameters including for the network,
      the optimizer and the batch size''')
    st.markdown("Default chosen parameters are the one used in the PixelCNN original publication")
    st.markdown('''Warning: choosing a high number of residual block , epochs or a  large number for h may result in a larger computation time.
      The batch size will also impact computation time.''')
    
    sel_col, disp_col = st.columns(2)
    if "h_channels_user" not in st.session_state:
        st.session_state["h_channels_user"] = ""
    
    if "epoch_user" not in st.session_state:
        st.session_state["epoch_user"] = ""
    
    if "nb_layer_block_user" not in st.session_state:
        st.session_state["nb_layer_block_user"] = ""

    if "user_loss_fn" not in st.session_state:
        st.session_state["user_loss_fn"] = ""

    if "user_model_mnist" not in st.session_state:
        st.session_state["user_model_mnist"] = ""
    
    if "optimizer_choice" not in st.session_state:
        st.session_state["optimizer_choice"] = ""
    
    if "batch_size_user" not in st.session_state:
        st.session_state["batch_size_user"] = ""

    h_channels_user = st.slider("Choose the h parameter",min_value=2, max_value=256,value=32,step=2)
    st.session_state["h_channels_user"] = h_channels_user

    nb_layer_blocks_user = st.slider("Choose the number of residual block",min_value=1, max_value=40,value=12,step=1)
    st.session_state["nb_layer_block_user"] = nb_layer_blocks_user
    
    epoch_user = st.slider("Choose the number of epoch",min_value=1, max_value=30,value=15,step=1)
    st.session_state["epoch_user"] = epoch_user

    batch_size_user = st.slider("Choose a batch size for the training of the model",min_value=2, max_value=512,value=16,step=2)
    st.session_state["batch_size_user"] = batch_size_user

    optimizer_choice = st.selectbox("Choose an optimizer for the network", options=(["Adam","Adagrad", "Adamax","RMSprop","AdaDelta"]), index = 3)
    st.session_state["optimizer_choice"] = optimizer_choice

    user_model_mnist = PixelCNN_MNIST(h_channels=h_channels_user, nb_layer_block = nb_layer_blocks_user)
    st.session_state["user_model_mnist"] = user_model_mnist

    user_loss_fn = nn.BCEWithLogitsLoss()
    st.session_state["user_loss_fn"] = user_loss_fn

    if "user_optimizer" not in st.session_state:
        st.session_state["user_optimizer"] = ""

    if "lr" not in st.session_state:
        st.session_state["lr"] = ""

    if optimizer_choice == "RMSprop":
        lr = st.selectbox("Choose a value for the learning rate", options=[0.00001, 0.0001, 0.001, 0.01, 0.1], index = 2)

        alpha =st.slider("Chose a value for the alpha parameter", min_value = 0.5, max_value=0.99,value=0.9, step = 0.01) 
        
        user_optimizer = torch.optim.RMSprop(user_model_mnist.parameters(),lr=lr,alpha=alpha)
        st.session_state["user_optimizer"] = user_optimizer

    elif optimizer_choice == "Adagrad":
        lr = st.selectbox("Choose a value for the learning rate", options=[0.00001, 0.0001, 0.001, 0.01, 0.1], index = 2)
        st.markdown("The following parameters are optional")
        lr_decay = st.selectbox("Choose a value for the learning rate decay", options=[0.00001, 0.0001, 0.001, 0.01,0.1,0], index = 5)
        weight_decay = st.selectbox("choose a value for the weight decay", options=[0.00001, 0.0001, 0.001, 0.01,0.1,0], index = 5)
        user_optimizer = torch.optim.Adagrad(user_model_mnist.parameters(),lr=lr,lr_decay=lr_decay, weight_decay=weight_decay)
        st.session_state["user_optimizer"] = user_optimizer

    elif optimizer_choice == "AdaDelta":
        lr = st.selectbox("Choose a value for the learning rate", options=[0.00001, 0.0001, 0.001, 0.01, 0.1], index = 2)
        
        st.markdown("The following parameters are optional")
        
        rho = st.slider("Choose a value for the rho parameter", min_value = 0.5, max_value=0.99,value=0.9, step = 0.01)
        weight_decay = st.selectbox("choose a value for the weight decay", options=[0.00001, 0.0001, 0.001, 0.01,0.1,0], index = 5)
        user_optimizer = torch.optim.Adadelta(user_model_mnist.parameters(),lr=lr,rho=rho, weight_decay= weight_decay)
        st.session_state["user_optimizer"] = user_optimizer


    elif optimizer_choice == "Adam":
        lr = st.selectbox("Choose a value for the learning rate", options=[0.00001, 0.0001, 0.001, 0.01, 0.1], index = 2)
        st.markdown("The following parameter is optional")
        weight_decay = st.selectbox("choose a value for the weight decay", options=[0.00001, 0.0001, 0.001, 0.01,0.1,0], index = 5)
        user_optimizer = torch.optim.Adam(user_model_mnist.parameters(),lr=lr, weight_decay= weight_decay)
        st.session_state["user_optimizer"] = user_optimizer
    else:
        lr = st.selectbox("Choose a value for the learning rate", options=[0.00001, 0.0001, 0.001, 0.01, 0.1], index = 2)
        st.markdown("The following parameter is optional")
        weight_decay = st.selectbox("choose a value for the weight decay", options=[0.00001, 0.0001, 0.001, 0.01,0.1,0], index = 5)
        user_optimizer = torch.optim.Adamax(user_model_mnist.parameters(),lr=lr, weight_decay= weight_decay)
        st.session_state["user_optimizer"] = user_optimizer

