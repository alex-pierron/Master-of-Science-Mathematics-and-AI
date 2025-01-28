from utils import *


def train_mnist(model, train_loader, val_loader,
                optimizer,  criterion = nn.MSELoss(), 
                num_epochs = 10, save_path = 'saved_models/mnist_model',
                print_logs = True, one_model = True):
    """
    Entraîne un modèle de réseau de neurones sur des données MNIST.

    Paramètres :
    - model (torch.nn.Module) : Le modèle à entraîner.
    - train_loader (torch.utils.data.DataLoader) : DataLoader pour les données d'entraînement.
    - val_loader (torch.utils.data.DataLoader) : DataLoader pour les données de validation.
    - optimizer (torch.optim.Optimizer) : Optimiseur utilisé pour la mise à jour des poids.
    - criterion (torch.nn.Module) : Fonction de perte utilisée pour calculer l'erreur.
    - num_epochs (int) : Nombre d'époques d'entraînement.
    - save_path (str) : Chemin pour sauvegarder les poids du modèle et les métriques.
    - print_logs (bool) : Afficher ou non les journaux pendant l'entraînement.
    - one_model (bool) : Utiliser ou non le backend matplotlib_inline pour l'affichage des graphiques.

    Renvoie :
    - metrics (dict) : Dictionnaire contenant les métriques d'entraînement.
    - checkpoint_path (str) : Chemin vers le point de contrôle du modèle après l'entraînement.
    """
    
    if one_model:
        plt.close('all')
        use("module://matplotlib_inline.backend_inline")
    else:
        plt.close('all')
        use('Agg')
    
    metrics = {'train_loss': [], 'train_accuracy': [], 'val_accuracy': []}
    val_accuracy_list = []
    val_accuracy = evaluate_mnist(model, val_loader, criterion,print_logs)
    validation_loss_list = []
    # Vérification et création du dossier de sauvegarde
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("Training ----------------------- Training ")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        correct_predictions_train = 0
        total_samples_train = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.view(-1, 28*28).float()
            labels = labels.float()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))

            # Backward pass et mise à jour des poids
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calcul de la précision d'entraînement
            predictions_train = torch.sign(outputs)
            correct_predictions_train += torch.sum(predictions_train == labels.view(-1, 1)).item()
            total_samples_train += len(labels)

        average_loss = total_loss / len(train_loader)
        metrics['train_loss'].append(average_loss)

        # Calcul de la précision d'entraînement
        train_accuracy = correct_predictions_train / total_samples_train * 100
        metrics['train_accuracy'].append(train_accuracy)
        if print_logs or epoch == num_epochs or epoch == 1:
            if print_logs:
                print(f'\n Epoch [{epoch}/{num_epochs}], Train Loss: {average_loss:.4f}, Train Accuracy: {train_accuracy:.4f} %')
            else:
                print(f' Epoch [{epoch}/{num_epochs}], Train Loss: {average_loss:.4f}, Train Accuracy: {train_accuracy:.4f} %')

        # Évaluation sur les données de validation
        if epoch == num_epochs:
            print("Evaluating ----------- Evaluating")
            val_accuracy,val_loss = evaluate_mnist(model, val_loader, criterion)
        else:
            val_accuracy,val_loss = evaluate_mnist(model, val_loader, criterion,print_logs=print_logs)
        metrics['val_accuracy'].append(val_accuracy)
        val_accuracy_list.append(val_accuracy)
        validation_loss_list.append(val_loss)
        # Sauvegarde des poids du modèle et des métriques avec suffixe d'epoch
        checkpoint_path = os.path.join(save_path, f'mnist_epoch_{epoch}.pth')
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }
        torch.save(checkpoint, checkpoint_path)

    # Affichage du subplot avec l'évolution des losses et des taux de classification
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot Loss
    axes[0].plot(range(1,num_epochs+1),metrics['train_loss'], label='Train Loss')
    axes[0].plot(range(1,num_epochs+1),validation_loss_list, label='Validation Loss', linestyle='--')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Plot Accuracy
    axes[1].plot(range(1,num_epochs+1),metrics['train_accuracy'], label='Train Accuracy')
    axes[1].plot(range(1,num_epochs+1),val_accuracy_list, label='Validation Accuracy', linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    # Sauvegarde du subplot dans le dossier de sauvegarde
    plot_path = os.path.join(save_path, 'training_plot.png')
    plt.savefig(plot_path)
    return metrics, checkpoint_path


def evaluate_mnist(model, val_loader, criterion, print_logs = True):
    """
    Évalue un modèle de réseau de neurones sur des données de validation MNIST.

    Paramètres :
    - model (torch.nn.Module) : Le modèle à évaluer.
    - val_loader (torch.utils.data.DataLoader) : DataLoader pour les données de validation.
    - criterion (torch.nn.Module) : Fonction de perte utilisée pour calculer l'erreur.
    - print_logs (bool) : Afficher ou non les journaux pendant l'évaluation.

    Renvoie :
    - accuracy (float) : Précision du modèle sur les données de validation.
    - average_loss (float) : Perte moyenne sur les données de validation.
    """
    model.eval()
    correct_predictions = 0
    total_samples = 0
    total_loss = 0

    with torch.no_grad():
        for inputs, labels in val_loader:

            inputs = inputs.view(-1, 28*28).float()

            # Conversion des données en torch.Tensor
            labels = labels.float()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))

            total_loss += loss.item()
            # Appliquer une fonction de signe pour obtenir -1 ou 1
            predictions = torch.sign(outputs)
            # Calcul de la précision
            correct_predictions += torch.sum(predictions == labels.view(-1, 1)).item()
            total_samples += len(labels)

        accuracy = correct_predictions / total_samples * 100
        average_loss = total_loss / len(val_loader)
        if print_logs:
            print(f'Validation Accuracy: {accuracy:.4f} %')
            print(f'Validation Loss: {average_loss:.4f}')
    return accuracy, average_loss
    


def calculate_2_layer_bound_mnist(model,checkpoint_path,
                                  gamma, delta, train_loader, print_norm = False):
    """
    Calcule la borne pour un modèle à deux couches sur des données MNIST.

    Paramètres :
    - model (torch.nn.Module) : Le modèle dont la borne doit être calculée.
    - checkpoint_path (str) : Chemin vers le point de contrôle (checkpoint) contenant les poids du modèle.
    - gamma (float) : Seuil pour la classification des échantillons.
    - delta (float) : Paramètre de la borne.
    - train_loader (torch.utils.data.DataLoader) : DataLoader pour les données d'entraînement.
    - print_norm (bool) : Afficher ou non la norme L1 des poids du modèle.

    Renvoie :
    - bound (float) : Borne calculée pour le modèle.
    """

    weight_norm = calculate_l1_norm(model,checkpoint_path, print_norm = print_norm)
    total_samples = 0
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        err_gamma_z = 0
        for inputs, labels in train_loader:

            inputs = inputs.view(-1, 28*28).float()
            labels = labels.float()
            
            outputs = model(inputs)

            err_gamma_z += torch.sum(outputs * labels.view(-1, 1) < gamma).item()
            total_samples += len(labels)
    
    err_gamma_z = err_gamma_z / total_samples
    c = 64 * np.log(6) /np.log(2)
    bound = err_gamma_z + np.sqrt( ( ( weight_norm**2 * 784 / (gamma**2) ) *
                                     np.log10(weight_norm / gamma) * np.log10(total_samples)**2 +
                                     np.log10(1/delta)) * ( c / total_samples ) )
    
    return bound