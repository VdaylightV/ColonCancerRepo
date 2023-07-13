import os
import torchio
from sklearn.model_selection import StratifiedShuffleSplit
import joblib

def main(args):
    """this is the main function that you should call

    Args:
        args (any): the args that train the model
        kernel (str): the kernel the SVM will use
        seed (int, optional): rand_seed. Defaults to 42.

    Returns:
        accuracy, precision, recall, f1 on validation
    """

    root_path = args.root_path
    num = args.num
    seed = args.seed
    C = args.C
    kernel = args.kernel
    target_shape = args.target_shape
    
    test2train_cwssim_matrix_savepath = args.test2train_cwssim_matrix_savepath
    dir_path = os.path.dirname(test2train_cwssim_matrix_savepath)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    image_files = []
    labels = []

    for subdir in os.listdir(root_path):
        if subdir.startswith('T'):
            if subdir == 'T3' or subdir == 'T4':
                for subject_dir in os.listdir(os.path.join(root_path, subdir))[:num]:
                    label = 0 if subdir in ['T1', 'T2'] else 1
                    image_path = os.path.join(root_path, subdir, subject_dir, "img_cut.nii.gz")
                    image_files.append(image_path)
                    labels.append(label)
            else:
                for subject_dir in os.listdir(os.path.join(root_path, subdir)):
                    label = 0 if subdir in ['T1', 'T2'] else 1
                    image_path = os.path.join(root_path, subdir, subject_dir, "img_cut.nii.gz")
                    image_files.append(image_path)
                    labels.append(label)

    # Define the torchio image transform to crop or pad images to (192, 192, 48)
    image_transform = torchio.Compose([
        torchio.CropOrPad(target_shape),
        torchio.RescaleIntensity(out_min_max=(0, 1)),
    ])

    # Use stratified train test split to divide the data into training and testing sets
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_indices, test_indices = next(splitter.split(image_files, labels))

    import nibabel as nib

    training_data = []
    for i in train_indices:
        obj = torchio.Subject(
            img=torchio.ScalarImage(path=image_files[i]),
            label=labels[i],
            transform=image_transform
        )
        training_data.append(image_transform(obj))

    single_test_image_obj = torchio.Subject(
            img=torchio.ScalarImage(path=args.image_path),
            label=args.image_label,
            transform=image_transform
        )
    
    testing_data = []

    testing_data.append(image_transform(single_test_image_obj))


    import numpy as np
    import torch
    from skimage.metrics import structural_similarity as ssim
    from pytorch_wavelets import DWTForward

    # Define the complex wavelet structural similarity function
    def cw_ssim(c_a, c_b, K=0):
        numerator = 2 * np.sum(c_a * np.conj(c_b)) + K
        denominator = np.sum(np.abs(c_a) ** 2) + np.sum(np.abs(c_b) ** 2) + K
        return numerator / denominator
       
            
    # Calculate the CW-SSIM matrix between all pairs of images in the testing set and the training set
    dwt = DWTForward(J=1, wave='db4')
    cw_ssim_matrix_testing_training = np.zeros((len(testing_data), len(training_data)))
    for i in range(len(testing_data)):
        image_i = testing_data[i]['img'][torchio.DATA].squeeze().double().numpy()
        coeffs_i = dwt(torch.from_numpy(image_i).unsqueeze(0).to(torch.float32))[0].squeeze().numpy()
        for j in range(len(training_data)):
            image_j = training_data[j]['img'][torchio.DATA].squeeze().double().numpy()
            coeffs_j = dwt(torch.from_numpy(image_j).unsqueeze(0).to(torch.float32))[0].squeeze().numpy()
            cw_ssim_ij = cw_ssim(coeffs_i, coeffs_j)
            cw_ssim_matrix_testing_training[i, j] = cw_ssim_ij

    with open(test2train_cwssim_matrix_savepath, 'wb') as f:
        np.save(f, cw_ssim_matrix_testing_training)
            
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler

    # Define the autoencoder model
    class Autoencoder(nn.Module):
        def __init__(self, input_size, encoding_size):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, encoding_size),
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_size, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, input_size),
            )

        def forward(self, x):
            x = x.view(x.size(0), -1)
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded

    # Set the embedding dimension
    embedding_dim = args.embedding_dim

    # Train the autoencoder on the CW-SSIM matrix between training images
    autoencoder = Autoencoder(len(training_data), embedding_dim)
    state_dict = torch.load(args.ae_model_savepath)
    autoencoder.load_state_dict(state_dict)

    normalizer = MinMaxScaler()
    cw_ssim_matrix_testing_training_scaled = normalizer.fit_transform(cw_ssim_matrix_testing_training)

    testing_embeddings, _ = autoencoder(torch.from_numpy(cw_ssim_matrix_testing_training_scaled).float())
    
    from sklearn import svm
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    # Train an SVM on the embeddings of the training set
    svm_model = joblib.load(args.model_savepath)

    # Make predictions on test set
    y_pred = svm_model.predict(testing_embeddings.detach().numpy())
    test_labels = np.array([single_test_image_obj.label])

    # Calculate evaluation metrics
    accuracy = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred)
    recall = recall_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred)
    cm = confusion_matrix(test_labels, y_pred)

    # Print evaluation metrics
    print("="*20, "TEST", "="*20)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Confusion matrix:\n{cm}")
    
    return accuracy, precision, recall, f1
    
if __name__ == "__main__":
    
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description='accept the config file path')

    parser.add_argument('config', type=str, help='path to config file')

    args = parser.parse_args()
    
    config_file_path = args.config
    config = OmegaConf.load(config_file_path)
    main(config)    


    
  


        

