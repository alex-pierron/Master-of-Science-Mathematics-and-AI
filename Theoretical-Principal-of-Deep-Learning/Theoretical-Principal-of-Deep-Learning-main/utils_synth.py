import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from utils import *
from sklearn.model_selection import train_test_split

def generate_synthetic_data(N, mean_1, std_1, mean_2, std_2, num_samples, seed = 42):
    """
    Génère des données synthétiques pour deux classes en utilisant des distributions normales.

    Paramètres :
    - N (int) : Dimensionnalité des données.
    - mean_1 (float) : Moyenne pour la classe 1.
    - std_1 (float) : Écart type pour la classe 1.
    - mean_2 (float) : Moyenne pour la classe 2.
    - std_2 (float) : Écart type pour la classe 2.
    - num_samples (int) : Nombre total d'échantillons à générer.

    Renvoie :
    - data (torch.Tensor) : Données synthétiques générées.
    - labels (torch.Tensor) : Étiquettes correspondantes aux classes (-1 pour la classe 1, 1 pour la classe 2).
    """
    torch.manual_seed(42)
    # Génération des données
    data_class_1 = torch.normal(mean=mean_1, std=std_1, size=(num_samples // 2, N))
    data_class_2 = torch.normal(mean=mean_2, std=std_2, size=(num_samples // 2, N))

    # Assignation des étiquettes
    labels_class_1 = torch.ones((num_samples // 2, 1)) * -1  # Étiquettes -1 pour la classe 1
    labels_class_2 = torch.ones((num_samples // 2, 1))  # Étiquettes 1 pour la classe 2

    # Concaténation des données et des étiquettes
    data = torch.cat((data_class_1, data_class_2), dim=0)
    labels = torch.cat((labels_class_1, labels_class_2), dim=0)

    return data, labels

# Création d'un DataLoader
class SyntheticDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

def add_noise(data, noise_std):
    """
    Ajoute du bruit gaussien à un ensemble de données.

    Args:
        data (torch.Tensor): Les données à ajouter du bruit.
        noise_std (float): L'écart-type (standard deviation) du bruit gaussien.

    Returns:
        torch.Tensor: Les données d'entrée avec du bruit ajouté.

    Exemple d'utilisation:
        >>> import torch
        >>> data = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        >>> noisy_data = add_noise(data, 0.1)
    """
    noise = torch.randn_like(data) * noise_std
    noisy_data = data + noise
    return noisy_data


def train_synth(model, train_loader, val_loader, optimizer,
                 criterion= nn.MSELoss(), num_epochs=10,
                 save_path='saved_models/synth_data',
                 print_logs =True, one_model=True):
    """
    Entraîne un modèle de réseau de neurones avec des données synthétiques.

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
    """
    
    if one_model:
        use("module://matplotlib_inline.backend_inline")
    else:
        plt.close('all')
        use('Agg')

    metrics = {'train_loss': [], 'train_accuracy': [], 'val_accuracy': []}
    validation_loss_list = []
    val_accuracy_list = []
    val_accuracy = evaluate_synth(model, val_loader, criterion,print_logs)
    # Vérification et création du dossier de sauvegarde
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("Training ----------------------- Training ")
    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0
        correct_predictions_train = 0
        total_samples_train = 0

        for inputs, labels in train_loader:
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
            val_accuracy,val_loss = evaluate_synth(model, val_loader, criterion)
        else:
            val_accuracy,val_loss = evaluate_synth(model, val_loader, criterion,print_logs=print_logs)
        val_accuracy_list.append(val_accuracy)
        validation_loss_list.append(val_loss)
        metrics['val_accuracy'].append(val_accuracy)

        # Sauvegarde des poids du modèle et des métriques avec suffixe d'epoch
        checkpoint_path = os.path.join(save_path, f'synth_data_epoch_{epoch+1}.pth')
        checkpoint = {
            'epoch': epoch ,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }
        torch.save(checkpoint, checkpoint_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot Loss
    axes[0].plot(metrics['train_loss'], label='Train Loss')
    axes[0].plot(validation_loss_list, label='Validation Loss', linestyle='--')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Plot Accuracy
    axes[1].plot(metrics['train_accuracy'], label='Train Accuracy')
    axes[1].plot(val_accuracy_list, label='Validation Accuracy', linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    # Sauvegarde du subplot dans le dossier de sauvegarde
    plot_path = os.path.join(save_path, 'training_plot.png')
    plt.savefig(plot_path)

    return metrics,checkpoint_path


def evaluate_synth(model, val_loader,criterion,print_logs=True):
    """
    Évalue un modèle de réseau de neurones avec des données de validation synthétiques.

    Paramètres :
    - model (torch.nn.Module) : Le modèle à évaluer.
    - val_loader (torch.utils.data.DataLoader) : DataLoader pour les données de validation.
    - criterion (torch.nn.Module) : Fonction de perte utilisée pour calculer l'erreur.
    - print_logs (bool) : Afficher ou non les journaux pendant l'évaluation.

    Renvoie :
    - accuracy (float) : Précision du modèle sur les données de validation.
    - average_mse (float) : Perte moyenne (MSE) sur les données de validation.
    """
    
    model.eval()
    correct_predictions = 0
    total_samples = 0
    total_mse = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            # Conversion des données en torch.Tensor
            labels = labels.float()

            # Forward pass
            outputs = model(inputs)
            mse = criterion(outputs, labels.view(-1, 1))

            total_mse += mse.item()
            # Appliquer une fonction de signe pour obtenir -1 ou 1
            predictions = torch.sign(outputs)
            # Calcul de la précision
            correct_predictions += torch.sum(predictions == labels.view(-1, 1)).item()
            total_samples += len(labels)

        accuracy = correct_predictions / total_samples * 100
        average_mse = total_mse / len(val_loader)
        if print_logs:
            print(f'Validation Accuracy: {accuracy:.4f} %')
            print(f'Validation Loss: {average_mse:.4f}')
    return accuracy, average_mse



def calculate_2_layer_bound_synth(model, N, checkpoint_path,
                                  gamma, delta, train_loader, print_norm = False):

    weight_norm = calculate_l1_norm(model,checkpoint_path, print_norm = print_norm)
    total_samples = 0
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        err_gamma_z = 0
        for inputs, labels in train_loader:

            labels = labels.float()
            
            outputs = model(inputs)

            err_gamma_z += torch.sum(outputs * labels.view(-1, 1) < gamma).item()
            total_samples += len(labels)
    
    err_gamma_z = err_gamma_z / total_samples
    c = 64 * np.log(6) /np.log(2)
    bound = err_gamma_z + np.sqrt( ( ( weight_norm**2 * N / (gamma**2) ) *
                                     np.log10(weight_norm / gamma) * np.log10(total_samples)**2 +
                                     np.log10(1/delta)) * (c / total_samples ))
    
    return bound


def comparatif_synth_2_layers( N , list_activation_function,
                              list_hidden_sizes, param_initialisation,
                              train_loader , val_loader , 
                              seed = 42, 
                              liste_delta = [0.05], liste_gamma = [0.5],
                              lr = 0.0001, num_epochs = 30,
                              try_number = 1):
    """
    Compare deux modèles de réseaux de neurones à deux couches cachées en utilisant des données synthétiques.

    Paramètres :
    - N (int) : Dimensionnalité des données d'entrée.
    - list_activation_function (list) : Liste de fonctions d'activation pour les couches cachées.
    - list_hidden_sizes (list) : Liste de deux listes contenant les tailles des couches cachées pour le Modèle A et le Modèle B.
    - param_initialisation (list) : Liste de deux listes contenant des tuples représentant la moyenne et l'écart type
      pour l'initialisation des poids pour le Modèle A et le Modèle B.
    - seed (int) : Graine pour la génération de nombres aléatoires.
    - liste_delta (list) : Liste des valeurs de delta pour le calcul de la borne.
    - liste_gamma (list) : Liste des valeurs de gamma pour le calcul de la borne.
    - lr (float) : Taux d'apprentissage pour l'optimiseur (SGD).
    - num_epochs (int) : Nombre d'époques d'entraînement.
    - train_loader (DataLoader) : DataLoader pour les données d'entraînement.
    - val_loader (DataLoader) : DataLoader pour les données de validation.
    - try_number (int) : Identifiant de l'expérience ou de la comparaison.

    Renvoie :
    - df_A (DataFrame) : DataFrame contenant les résultats et les métriques pour le Modèle A.
    - df_B (DataFrame) : DataFrame contenant les résultats et les métriques pour le Modèle B.
    """
    
    activation_function_names = [str(activation_fn).split("(")[0] for activation_fn in list_activation_function]

    columns_A = ['num_epochs', 'activation_function', 'hidden_size', 'param_mean', 'param_std', 'delta', 'gamma','train_accuracy','train loss', 'validation_accuracy','validation_loss', 'bound']
    df_A = pd.DataFrame(columns=columns_A)

    data_A = []

    columns_B = ['num_epochs', 'activation_function', 'hidden_size', 'param_mean', 'param_std', 'delta', 'gamma','train_accuracy','train_loss', 'validation_accuracy','validation_loss', 'bound']
    df_B = pd.DataFrame(columns=columns_B)
    data_B = []
    
    for index_activation, activation_function in enumerate(list_activation_function):
        for index_size, hidden_size in enumerate(list_hidden_sizes[0]):
            for index_param, param in enumerate(param_initialisation[0]):

                model_A_synth = MLP(N, hidden_size,activation_function = activation_function)

                initialize_weights(model_A_synth, param[0], param[1],seed = seed)

                checkpoint_path = f'saved_models/synth_model/compare_2_layers/{try_number}/model_A/activation_{activation_function_names[index_activation]}/hidden_size_{hidden_size[0]}/mean_{param[0]}_std_{param[1]}'

                optimizer = optim.SGD(model_A_synth.parameters(), lr=lr)
                metrics_A, checkpoint_path = train_synth(model_A_synth, train_loader, val_loader,
                                    optimizer, num_epochs=num_epochs,
                                    save_path= checkpoint_path,
                                    print_logs=False, one_model=False)
                
                train_accuracy, train_loss = evaluate_synth(model_A_synth, train_loader, nn.MSELoss(),print_logs = False)
                validation_accuracy, validation_loss = evaluate_synth(model_A_synth, val_loader, nn.MSELoss(),print_logs = False)
                        
                for index_delta, delta in enumerate(liste_delta):

                    for index_gamma, gamma in enumerate(liste_gamma):
                        
                        data_A.append({
                            'num_epochs': num_epochs,
                            'activation_function': activation_function_names[index_activation],
                            'hidden_size': hidden_size,
                            'param_mean': param[0],
                            'param_std': param[1],
                            'delta': delta,
                            'gamma': gamma,
                            'train_accuracy': train_accuracy,
                            'train_loss': train_loss,
                            'validation_accuracy': validation_accuracy,
                            'validation_loss': validation_loss,
                            'bound': calculate_2_layer_bound_synth(model_A_synth,N,checkpoint_path,gamma,delta,train_loader)
                        })

        for index_size, hidden_size in enumerate(list_hidden_sizes[1]):
            for index_param, param in enumerate(param_initialisation[1]):

                model_B = MLP(N, hidden_size,activation_function = activation_function)

                initialize_weights(model_B, param[0], param[1],seed = seed)

                checkpoint_path = f'saved_models/synth_model/compare_2_layers/{try_number}/model_B/activation_{activation_function_names[index_activation]}/hidden_size_{hidden_size[0]}/mean_{param[0]}_std_{param[1]}'

                optimizer = optim.SGD(model_B.parameters(), lr=lr)
                metrics_B,checkpoint_path = train_synth(model_B, train_loader, val_loader,
                                    optimizer, num_epochs=num_epochs,
                                    save_path = checkpoint_path,
                                    print_logs=False, one_model = False)

                train_accuracy, train_loss = evaluate_synth(model_B, train_loader, nn.MSELoss(),print_logs = False)
                validation_accuracy, validation_loss = evaluate_synth(model_B, val_loader, nn.MSELoss(),print_logs = False)
                        
                for index_delta, delta in enumerate(liste_delta):
                    for index_gamma, gamma in enumerate(liste_gamma):
                        data_B.append({
                            'num_epochs': num_epochs,
                            'activation_function': activation_function_names[index_activation],
                            'hidden_size': hidden_size,
                            'param_mean': param[0],
                            'param_std': param[1],
                            'delta': delta,
                            'gamma': gamma,
                            'train_accuracy': train_accuracy,
                            'train_loss': train_loss,
                            'validation_accuracy': validation_accuracy,
                            'validation_loss': validation_loss,
                            'bound': calculate_2_layer_bound_synth(model_B,N,checkpoint_path,gamma,delta,train_loader)
                        })
    df_A = pd.concat([df_A, pd.DataFrame(data_A)], ignore_index=True)

    df_B = pd.concat([df_B, pd.DataFrame(data_B)], ignore_index=True)

    # Sauvegarde du DataFrame A
    df_A.to_csv(f'saved_models/synth_model/compare_2_layers/{try_number}/model_A/df_A.csv', index=False)

    # Sauvegarde du DataFrame B
    df_B.to_csv(f'saved_models/synth_model/compare_2_layers/{try_number}/model_B/df_B.csv', index=False)
    
    return df_A, df_B