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
    
    train2train_cwssim_matrix_savepath = args.train2train_cwssim_matrix_savepath
    dir_path = os.path.dirname(train2train_cwssim_matrix_savepath)[0]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        
    test2train_cwssim_matrix_savepath = args.test2train_cwssim_matrix_savepath
    dir_path = os.path.dirname(test2train_cwssim_matrix_savepath)[0]
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

    testing_data = []
    for i in test_indices:
        obj = torchio.Subject(
            img=torchio.ScalarImage(path=image_files[i]),
            label=labels[i],
            transform=image_transform
        )
        testing_data.append(image_transform(obj))


    import numpy as np
    import torch
    from skimage.metrics import structural_similarity as ssim
    from pytorch_wavelets import DWTForward

    # Define the complex wavelet structural similarity function
    def cw_ssim(c_a, c_b, K=0):
        numerator = 2 * np.sum(c_a * np.conj(c_b)) + K
        denominator = np.sum(np.abs(c_a) ** 2) + np.sum(np.abs(c_b) ** 2) + K
        return numerator / denominator

    # Calculate the CW-SSIM matrix between all pairs of images in the training set
    dwt = DWTForward(J=1, wave='db4')
    cw_ssim_matrix_training_training = np.zeros((len(training_data), len(training_data)))
    for i in range(len(training_data)):
        image_i = training_data[i]['img'][torchio.DATA].squeeze().double().numpy()
        coeffs_i = dwt(torch.from_numpy(image_i).unsqueeze(0).to(torch.float32))[0].squeeze().numpy()
        for j in range(i, len(training_data)):
            image_j = training_data[j]['img'][torchio.DATA].squeeze().double().numpy()
            coeffs_j = dwt(torch.from_numpy(image_j).unsqueeze(0).to(torch.float32))[0].squeeze().numpy()
            cw_ssim_ij = cw_ssim(coeffs_i, coeffs_j)
            cw_ssim_matrix_training_training[i, j] = cw_ssim_ij
            cw_ssim_matrix_training_training[j, i] = cw_ssim_ij

    with open(train2train_cwssim_matrix_savepath, 'wb') as f:
        np.save(f, cw_ssim_matrix_training_training)
        
    # with open(train2train_cwssim_matrix_savepath, 'rb') as f:
    #     cw_ssim_matrix_training_training = np.load(f)
    #     print(cw_ssim_matrix_training_training.shape)
        
            
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
        
    # with open("testing2traing_cwssim_matrix.npy", 'rb') as f:
    #     cw_ssim_matrix_testing_training = np.load(f)
    #     print(cw_ssim_matrix_testing_training.shape)

            
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
    cw_ssim_matrix_training_training_scaled = MinMaxScaler().fit_transform(cw_ssim_matrix_training_training)
    autoencoder = Autoencoder(len(training_data), embedding_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)

    num_epochs = args.epochs
    for epoch in range(num_epochs):
        for i in range(len(training_data)):
            input_data = torch.from_numpy(cw_ssim_matrix_training_training_scaled[i]).unsqueeze(0).float()
            target_data = input_data
            encoded, decoded = autoencoder(input_data)

            loss = criterion(decoded, target_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    dir_path = os.path.dirname(args.ae_model_savepath)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    state_dict = autoencoder.state_dict()
    torch.save(state_dict, args.ae_model_savepath)
    
    # Generate the embeddings for training and testing sets
    normalizer = MinMaxScaler()
    cw_ssim_matrix_training_training_scaled = normalizer.fit_transform(cw_ssim_matrix_training_training)
    cw_ssim_matrix_testing_training_scaled = normalizer.transform(cw_ssim_matrix_testing_training)

    training_embeddings, _ = autoencoder(torch.from_numpy(cw_ssim_matrix_training_training_scaled).float())
    testing_embeddings, _ = autoencoder(torch.from_numpy(cw_ssim_matrix_testing_training_scaled).float())
    
    from sklearn import svm

    # Train an SVM on the embeddings of the training set
    svm_model = svm.SVC(kernel=kernel, C=C, random_state=seed)
    svm_model.fit(training_embeddings.detach().numpy(), np.array(labels)[train_indices])
    filename = args.model_savepath
    dir_path = os.path.dirname(args.model_savepath)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    joblib.dump(svm_model, filename)

    # Make predictions on test set
    y_pred = svm_model.predict(training_embeddings.detach().numpy())
    train_labels = np.array(labels)[train_indices]

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    # Calculate evaluation metrics
    accuracy = accuracy_score(train_labels, y_pred)
    precision = precision_score(train_labels, y_pred)
    recall = recall_score(train_labels, y_pred)
    f1 = f1_score(train_labels, y_pred)
    cm = confusion_matrix(train_labels, y_pred)

    # Print evaluation metrics
    print("="*20, "TRAIN", "="*20)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Confusion matrix:\n{cm}")

    ############################################################

    # Make predictions on test set
    y_pred = svm_model.predict(testing_embeddings.detach().numpy())
    test_labels = np.array(labels)[test_indices]

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

    print('Conducting visualization!')

    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # 读取数据

    # 将两个tensor合并为一个
    all_embeddings = np.concatenate([training_embeddings.detach().numpy(), testing_embeddings.detach().numpy()], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    
    print(all_labels)

    # 使用TSNE进行降维
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # 绘制图像
    plt.figure(figsize=(8, 8))
    markers = ['^', 's']  # 标记形状，三角形和五角星
    colors = ['r', 'b']   # 不同类别用不同颜色表示

    # 绘制训练集数据
    for i in range(len(colors)):
        marker = 'o'
        color = colors[i]
        plt.scatter(embeddings_2d[all_labels == i, 0], embeddings_2d[all_labels == i, 1], marker=marker, color=color, label='Class {}'.format(i), s=500, alpha=0.5)

    # 绘制测试集数据
    for i in range(len(colors)):
        marker = '^'  # 五角星形状
        color = colors[i]
        test_indices = np.where(test_labels == i)[0]  # 获取测试集中当前类别的索引
        plt.scatter(embeddings_2d[len(training_embeddings) + test_indices, 0], embeddings_2d[len(training_embeddings) + test_indices, 1], marker=marker, color=color, label='Class {} (Test)'.format(i), s=500)

    plt.title('Visualization')
    plt.legend(loc='upper right')
    dir_path = os.path.dirname(args.visualization_savepath)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    plt.savefig(args.visualization_savepath)
    
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


    
  


        

