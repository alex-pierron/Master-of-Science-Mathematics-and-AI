import streamlit as st
from MaskedConv import *
from PixelCNN_MNIST_model import PixelCNN_MNIST
from training_loop import train_loop
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets

training_model = st.container()



with training_model:
    st.title("Training your custom model")
    st.markdown('''Here you can train the model. Please note that is might be time consuming.
                During training you will have the mean loss for each epoch. When the training will 
                finish an aditionnal plot will show how the loss evolves ''')
    st.markdown("""Once you are have selected the desired parameters, you can train the model by selecting 'training mode' down below. 'Standby mode' is here by default to let you choose the parameters """)
    if st.button('Click here to begin training'):
        list_epoch = []
        best_loss = 1
        train_loader_user = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                        lambda x: x>0,
                        lambda x: x.float(),
                ])),batch_size=st.session_state["batch_size_user"], shuffle=True,pin_memory=True)
        
        for t in range(st.session_state["epoch_user"]):
            print(f"Epoch {t+1}\n-------------------------------")
            accuracy,best_loss = train_loop(train_loader_user,
                                             st.session_state["user_model_mnist"], 
                                             st.session_state["user_loss_fn"], 
                                             st.session_state["user_optimizer"],
                                             best_loss)
            list_epoch.append(accuracy)
            st.write(f"Mean loss for this epoch: {accuracy}")
            st.write(f"Minimal loss reached so far: {best_loss}")
        print("Done!")
        
        fig,ax = plt.subplots()
        ax.plot(range(st.session_state["epoch_user"]),list_epoch)
        plt.xlabel("Number of the epoch")
        plt.ylabel("Loss")
        plt.title("Evolution of the loss during training")
        st.pyplot(fig)

    st.subheader("Saving the model")

    st.markdown(''' If you are satisfied with the best accuracy achieved by the model, you can save it.
    Please note that you can save only one model at the time.
    The weights of the model will be saved inside the 'mnist_user_model.pth' file. ''')

    st.markdown(""" Your custom model you will usable for generating images in the next section.""")

    if st.button("Click here if you want to save the model"):
        if "PATH" not in st.session_state:
            st.session_state["PATH"] = ""
        PATH = 'training_weight/mnist_user_model.pth'
        st.session_state["PATH"] = PATH
        torch.save(st.session_state["user_model_mnist"].state_dict(), PATH)