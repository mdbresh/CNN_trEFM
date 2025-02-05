import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
from tqdm import tqdm

class SignalDataset(Dataset):
    def __init__(self, signals, parameters, parameter_ranges):
        """
        Initialize a SignalDataset object.

        Parameters
        ----------
        signals : numpy array
            A numpy array of signals to be processed.
        parameters : list or numpy array
            A numpy array of labeled parameters (true parameter values).
        parameter_ranges : list of tuples
            A list of tuples, where each tuple specifies the range of a parameter.
            For example, [(1, 10), (50, 500), (0.1, 0.9)].

        Returns
        -------
        A SignalDataset object.
        """
        self.signals = torch.FloatTensor(signals)
        # Normalize parameters to 0-1 range for better training
        self.parameters = torch.FloatTensor(parameters)
        self.param_ranges = parameter_ranges
        self.normalize_parameters()
       
    
    def normalize_parameters(self):
        """
        Normalize parameters to a 0-1 range based on parameter ranges.

        This function normalizes each parameter in `self.parameters` by scaling 
        the values to a 0-1 range using the corresponding range specified in 
        `self.param_ranges`. The normalization is performed by applying the 
        formula: 
            normalized_value = (value - min_val) / (max_val - min_val)

        The normalized parameters are stored back into `self.parameters`.

        Assumes that `self.param_ranges` contains a list of tuples, where each 
        tuple is of the form (min_val, max_val) for each parameter.

        Parameters are normalized to avoid order of magnitude errors during training.
        """
        normalized_params = []
        for i in range(3):
            min_val, max_val = self.param_ranges[i]
            normalized = (self.parameters[:, i] - min_val) / (max_val - min_val)
            normalized_params.append(normalized.unsqueeze(1))
        
        self.parameters = torch.cat(normalized_params, dim=1)
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        return self.signals[idx], self.parameters[idx]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Initialize a ResidualBlock object.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int
            Kernel size for the first convolutional layer.
        stride : int
            Stride for the first convolutional layer.
        padding : int
            Padding for the first convolutional layer.

        Notes
        -----
        The ResidualBlock consists of two convolutional layers with a LeakyReLU
        activation function after the first layer and before the second layer.
        The output of the second layer is added to the input to form the output
        of the ResidualBlock. A skip connection is added if the number of input
        channels and output channels are different or if the stride is not 1.
        """
        super(ResidualBlock, self).__init__()
        
        self.main_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        
        # Skip connection with 1x1 conv if dimensions change
        self.skip_connection = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride, 0),
            nn.BatchNorm1d(out_channels)
        ) if (in_channels != out_channels or stride != 1) else nn.Identity()
        
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        return self.activation(self.main_path(x) + self.skip_connection(x))

class SignalNet(nn.Module):
    def __init__(self, parameter_ranges, dropout_rate=0.2):
        """
        Initialize a SignalNet object.

        Parameters
        ----------
        parameter_ranges : list of tuples
            A list of tuples specifying the range of each parameter. For example, [(min1, max1), (min2, max2), ...].
        dropout_rate : float, optional
            The dropout rate applied to various layers in the network. Default is 0.2.

        Notes
        -----
        The SignalNet architecture includes three main branches:
        - Fast component branch: Extracts features from the first 2000 points of the input using small kernels.
        - Slow component branch: Analyzes the full signal with larger kernels to capture broader patterns.
        - Early features branch: Focuses on capturing initial features with increased strides and adjusted pooling.

        The extracted features from each branch are combined and processed through shared layers, followed by 
        separate heads for each parameter type, which produce predictions using sigmoid activation.
        """

        super(SignalNet, self).__init__()
        
         # Fast component branch - fast (small) kernel
        self.fast_branch = nn.Sequential(
            # Initial precise feature extraction
            nn.Conv1d(15000, 32, kernel_size=51, stride=2, padding=25),  # Output: ~7500
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            ResidualBlock(32, 64, kernel_size=25, stride=2, padding=12),  # Output: ~3750
            nn.Dropout(dropout_rate),
            
            ResidualBlock(64, 128, kernel_size=15, stride=2, padding=7),  # Output: ~1875
            nn.Dropout(dropout_rate), # output size 128
            
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Slow component branch - analyzes full signal with large kernels
        self.slow_branch = nn.Sequential(
            nn.Conv1d(15000, 32, kernel_size=251, stride=10, padding=125), 
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            ResidualBlock(32, 64, kernel_size=201, stride=5, padding=100), 
            nn.Dropout(dropout_rate),
            
            ResidualBlock(64, 128, kernel_size=101, stride=5, padding=50),
            nn.Dropout(dropout_rate),
            
            nn.AdaptiveAvgPool1d(1) # output size 128
        )
        
        # Early features branch - specifically for the initial drop
        self.early_features = nn.Sequential(
            nn.Conv1d(15000, 32, kernel_size=25, stride=5, padding=12),  # Increased stride
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2),  # Adjusted pooling params
            nn.Conv1d(32, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.AdaptiveMaxPool1d(1) # outputsize 32
        )
        
        # Combine features from all branches
        combined_size = 128 + 128 + 32 + 3  # fast + slow + early + cantilever parameters

        """
        self.shared_features is a dense NN that contextualizes the latent parameters
        learned by the branches with the cantilever parameters. This step introduces
        the cantilever physics into the learning process.
        """
        self.shared_features = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        
        # Separate heads for each parameter type
        # Each head regresses on the contexutalized latent parameters from self.shared_features
        self.fast_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.slow_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.weight_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.param_ranges = parameter_ranges
    
    def forward(self, x):
        c = x[:,:3]
        x = x[:,3:]
        x = x.unsqueeze(-1)
        
        # Process through specialized branches
        fast_features = self.fast_branch(x).squeeze(-1)
        slow_features = self.slow_branch(x).squeeze(-1)
        early_features = self.early_features(x).squeeze(-1)
        
        # Combine all features
        combined = torch.cat((fast_features, slow_features, early_features, c), dim=1)
        shared = self.shared_features(combined)
        
        # Generate predictions
        param1 = self.fast_head(shared)
        param2 = self.slow_head(shared)
        param3 = self.weight_head(shared)
        
        return torch.cat((param1, param2, param3), dim=1)


class WeightedMSELoss(nn.Module):
    def __init__(self, param_ranges):
        super(WeightedMSELoss, self).__init__()
        self.param_ranges = param_ranges
        
        # Calculate weights based on parameter ranges
        self.weights = [2.5, 1.75,1]
        # for min_val, max_val in param_ranges:
        #     range_size = max_val - min_val
        #     # Inverse range size for weight calculation
        #     self.weights.append(1.0 / (range_size ** 2))
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def forward(self, pred, target):
        losses = []
        for i, weight in enumerate(self.weights):
            mse = torch.mean((pred[:, i] - target[:, i]) ** 2)
            losses.append(weight * mse)
        return sum(losses)

def train_model(model, train_loader, val_loader, param_ranges, monitor, num_epochs=150):
    """
    Train a model for a given number of epochs.

    Parameters
    ----------
    model : nn.Module
        Model to be trained.
    train_loader : DataLoader
        Data loader for training data.
    val_loader : DataLoader
        Data loader for validation data.
    param_ranges : list of tuples
        List of tuples containing parameter ranges.
    monitor : TrainingMonitor
        Monitoring object for tracking training progress.
    num_epochs : int, optional
        Number of epochs to train for. Defaults to 150.

    Returns
    -------
    None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Weighted loss
    criterion = WeightedMSELoss(param_ranges)
    
    # One cycle learning rate policy
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Learning rate annealing using OneCycleLR
    # Changes how quickly the learning rate decays
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-1,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for signals, params in progress_bar:
            signals, params = signals.to(device), params.to(device)
            
            optimizer.zero_grad()
            outputs = model(signals)
            
            loss = criterion(outputs, params)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({
                'train_loss': f'{loss.item():.6f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad(): # don't update weights
            for signals, params in val_loader:
                signals, params = signals.to(device), params.to(device)
                outputs = model(signals)
                loss = criterion(outputs, params)
                val_loss += loss.item()
                val_predictions.append(outputs)
                val_targets.append(params)
        
        # Prepare validation predictions and targets for monitor
        val_predictions = torch.cat(val_predictions, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        val_loss = val_loss / len(val_loader)
        
        # Update monitor
        monitor.update(epoch, train_loss, val_loss, val_predictions, val_targets)
        
        # Early stopping with longer patience
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, f'{monitor.run_dir}/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after epoch {epoch+1}")
                break
    

class TrainingMonitor:
    def __init__(self, param_ranges):
        """
        Initialize a TrainingMonitor object.
        TrainingMonitor is used to track the progress of training and validation.
        It saves the loss values, parameter errors, and R² scores at each epoch.
        
        Parameters
        ----------
        param_ranges : list of tuples
            A list of tuples, where each tuple specifies the range of a parameter.
            For example, [(1, 10), (50, 500), (0.1, 0.9)].
        
        Attributes
        ----------
        param_ranges : list of tuples
            A copy of the input param_ranges.
        train_losses : list
            A list of the training loss values at each epoch.
        val_losses : list
            A list of the validation loss values at each epoch.
        param_errors : dict
            A dictionary of lists, where each list contains the mean absolute error
            of a parameter at each epoch.
        param_r2 : dict
            A dictionary of lists, where each list contains the R² score of a parameter
            at each epoch.
        start_time : datetime
            The time when training started.
        run_dir : str
            The directory where all output files are saved.
        
        """
       
        self.param_ranges = param_ranges
        self.train_losses = []
        self.val_losses = []
        self.param_errors = {i: [] for i in range(3)}
        self.param_r2 = {i: [] for i in range(3)}
        self.start_time = datetime.now()
        
        # Create output directory for this run
        self.run_dir = f"training_run_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.run_dir, exist_ok=True)
    
    def compute_r2(self, predictions, targets):
        """Compute R² score for a single parameter"""
        # Convert to numpy for easier calculation
        y_true = targets.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        
        # R² calculation
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))  # Add small epsilon to prevent division by zero
        return r2
    
    def rescale_params(self, params, param_idx):
        """Rescale normalized parameters back to original range"""
        min_val, max_val = self.param_ranges[param_idx]
        return params * (max_val - min_val) + min_val
    
    def update(self, epoch, train_loss, val_loss, val_predictions, val_targets):
        """
        Update training history and save progress.
        
        Parameters
        ----------
        epoch : int
            The current epoch number.
        train_loss : float
            The average loss on the training set at this epoch.
        val_loss : float
            The average loss on the validation set at this epoch.
        val_predictions : torch.tensor
            The predictions on the validation set at this epoch.
        val_targets : torch.tensor
            The true targets on the validation set at this epoch.
        
        Notes
        -----
        This function is called at the end of each epoch in the training loop.
        It updates the training history and saves the current state of the model
        to disk. It also creates plots of the training history every 5 epochs.
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # Calculate parameter-specific errors and R² scores
        for i in range(3):
            # Rescale predictions and targets to original range
            pred_rescaled = self.rescale_params(val_predictions[:, i], i)
            target_rescaled = self.rescale_params(val_targets[:, i], i)
            
            # Calculate MAE in original scale
            mae = torch.mean(torch.abs(pred_rescaled - target_rescaled)).item()
            self.param_errors[i].append(mae)
            
            # Calculate R² score
            r2 = self.compute_r2(pred_rescaled, target_rescaled)
            self.param_r2[i].append(r2)
        
        # Save current state
        self.save_progress()
        
        # Create visualizations
        if (epoch + 1) % 5 == 0:  # Create plots every 5 epochs
            self.create_plots(epoch + 1)
    
    def save_progress(self):
        """
        Save the current state of the training monitor to a JSON file.

        The saved file contains the following information:
        - Training loss values at each epoch
        - Validation loss values at each epoch
        - Parameter-specific mean absolute errors at each epoch
        - Parameter-specific R² scores at each epoch
        - Parameter ranges

        The file is saved in the run directory specified by `self.run_dir`.
        This function was adapted from StackOverflow.      
        """
        progress = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'param_errors': self.param_errors,
            'param_r2': self.param_r2,
            'param_ranges': self.param_ranges
        }
        
        with open(f'{self.run_dir}/progress.json', 'w') as f:
            json.dump(progress, f)
    
    def create_plots(self, epoch):
        """
        Create and save plots for training progress.

        This method generates three subplots displaying the training and validation loss,
        parameter-specific mean absolute errors (MAE), and parameter-specific R² scores
        over epochs. The plots are saved as a PNG file in the run directory.

        Parameters
        ----------
        epoch : int
            The current epoch number, used for labeling the plots and file name.
        """

        plt.figure(figsize=(15, 15))
        
        # Loss plot
        plt.subplot(3, 1, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title(f'Training Progress - Epoch {epoch}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Parameter errors plot
        plt.subplot(3, 1, 2)
        for i in range(3):
            plt.plot(self.param_errors[i], 
                    label=f'Param {i+1} MAE ({self.param_ranges[i][0]:.1f}-{self.param_ranges[i][1]:.1f})')
        plt.title('Parameter-Specific MAE (Original Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        plt.grid(True)
        
        # R² scores plot
        plt.subplot(3, 1, 3)
        for i in range(3):
            plt.plot(self.param_r2[i],
                    label=f'Param {i+1} R² ({self.param_ranges[i][0]:.1f}-{self.param_ranges[i][1]:.1f})')
        plt.title('Parameter-Specific R² Score')
        plt.xlabel('Epoch')
        plt.ylabel('R²')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.run_dir}/progress_epoch_{epoch}.png')
        plt.close()

def visualize_predictions(predictions, targets, param_ranges, save_dir):
    """
    Visualize the predictions against true target values for each parameter.

    This function generates scatter plots comparing predicted and true values for
    three parameters. Each subplot includes a perfect prediction line, R² score,
    mean absolute error (MAE), and root mean square error (RMSE) statistics.

    Parameters
    ----------
    predictions : torch.Tensor
        A tensor containing predicted values for each parameter.
    targets : torch.Tensor
        A tensor containing true values for each parameter.
    param_ranges : list of tuples
        A list of tuples where each tuple specifies the range of a parameter.
    save_dir : str
        The directory where the resulting plot will be saved.

    The plots are saved as 'prediction_scatter.png' in the specified save_dir.
    """

    plt.figure(figsize=(15, 5))
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        
        min_val, max_val = param_ranges[i]
        # Get the current parameter's predictions and targets
        y_pred = (predictions[:, i] * (max_val - min_val) + min_val).cpu().numpy()
        y_true = (targets[:, i] * (max_val - min_val) + min_val).cpu().numpy()
        # y_true = targets[:, i].cpu().numpy()
        # y_pred = predictions[:, i].cpu().numpy()
        
        # Calculate R² score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        plt.scatter(y_true, y_pred, alpha=0.5)
        min_val, max_val = param_ranges[i]
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # Perfect prediction line
        
        plt.title(f'Parameter {i+1} Predictions\nR² = {r2:.4f}')
        plt.xlabel('True Value')
        plt.ylabel('Predicted Value')
        
        # Set axis limits to parameter ranges
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        
        # Add textbox with statistics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        stats_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}'
        plt.text(0.05, 0.95, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/prediction_scatter.png')
    plt.close()

def evaluate_model(model, test_loader, device, param_ranges):
    """
    Evaluate the model on the test dataset and return predictions and targets.

    This function sets the model to evaluation mode and iterates over the test
    dataset to generate predictions for each batch of signals. Both predictions
    and target parameters are rescaled to their original ranges specified by 
    `param_ranges`.

    Parameters
    ----------
    model : nn.Module
        The trained model to be evaluated.
    test_loader : DataLoader
        Data loader for the test dataset.
    device : torch.device
        The device (CPU or GPU) on which to perform the evaluation.
    param_ranges : list of tuples
        A list of tuples where each tuple specifies the range of a parameter.

    Returns
    -------
    predictions : torch.Tensor
        A tensor containing the predicted values for the test dataset.
    targets : torch.Tensor
        A tensor containing the true parameter values for the test dataset.
    """

    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for signals, params in test_loader:
            signals, params = signals.to(device), params.to(device)
            outputs = model(signals)
            
            # Rescale predictions and targets to original ranges
            for i in range(3):
                min_val, max_val = param_ranges[i]
                outputs[:, i] = outputs[:, i] * (max_val - min_val) + min_val
                params[:, i] = params[:, i] * (max_val - min_val) + min_val
            
            predictions.append(outputs)
            targets.append(params)
    
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    
    return predictions, targets

def use_model(model, inputs, param_ranges):
    """
    Use the trained model to make predictions on new signals.

    This function takes in a trained model, a list of signals, and a list of
    parameter ranges, and returns predictions for each signal.

    Parameters
    ----------
    model : nn.Module
        The trained model to be used for prediction.
    inputs : list of torch.Tensor
        A list of tensors containing the input signals.
    param_ranges : list of tuples
        A list of tuples where each tuple specifies the range of a parameter.

    Returns
    -------
    predictions : array_like
        An array containing the predicted values for the input signals.
    """
    input_t = torch.from_numpy(inputs).float()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(input_t).detach().numpy()

        # Rescale predictions and targets to original ranges
        for i in range(3):
            min_val, max_val = param_ranges[i]
            outputs[:, i] = outputs[:, i] * (max_val - min_val) + min_val

    return outputs

def average_bi_tau(tau1, tau2, A):
    """
    Calculate the weighted average tau for a given output tensor.

    Parameters
    ----------
    output : numpy.ndarray
        An array containing the output of the model of shape `(n_samples, 3)`.

    Returns 
    -------
    avetau : numpy.ndarray
        An array containing the weighted average tau.
    """
    avetau = A*tau1 + (1-A)*tau2
    return avetau

def main(signals, parameters, batch_size=64, num_epochs=150):
    """
    Train a model on the given signals and parameters, and evaluate its performance on
    a test set.

    This function creates a dataset and data loaders for the given signals and
    parameters, splits them into training, validation, and test sets, and trains a
    model using the training set. The model is evaluated on the validation set
    during training, and its final performance is evaluated on the test set.

    Parameters
    ----------
    signals : array_like
        Array of shape `(n_samples, n_timesteps)` containing the input signals.
    parameters : array_like
        Array of shape `(n_samples, 3)` containing the true parameter values for
        each signal.
    batch_size : int, optional
        The batch size to use for training. Defaults to 64.
    num_epochs : int, optional
        The number of epochs to train the model for. Defaults to 150.

    Returns
    -------
    model : nn.Module
        The trained model.
    loaders : tuple of DataLoader
        A tuple containing the data loaders for the training, validation, and test
        sets.
    monitor : TrainingMonitor
        An instance of the TrainingMonitor class, which contains the training
        history and the directory where the model was saved.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Parameter ranges
    param_ranges = [(1.0, 10.0), (50.0, 500.0), (0.01, 0.99)]
    #param_ranges = [(1.0, 50.0), (100.0, 500.0), (0.1, 0.9)]
    
    # Initialize training monitor
    monitor = TrainingMonitor(param_ranges)
    
    # Create datasets and dataloaders
    dataset = SignalDataset(signals, parameters, parameter_ranges=param_ranges)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"Training on {len(train_dataset)} samples")
    print(f"Validating on {len(val_dataset)} samples")
    print(f"Testing on {len(test_dataset)} samples")
    
    # Create and train model
    model = SignalNet(parameter_ranges=param_ranges, dropout_rate=0.2)
    train_model(model, train_loader, val_loader, param_ranges, monitor, num_epochs=num_epochs)
    
    # Final evaluation on test set
    print("\nPerforming final evaluation on test set...")
    model.eval()
    test_predictions = []
    test_targets = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for signals, params in test_loader:
            signals, params = signals.to(device), params.to(device)
            outputs = model(signals)
            test_predictions.append(outputs)
            test_targets.append(params)
    
    test_predictions = torch.cat(test_predictions, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    
    # Calculate and display final metrics
    print("\nFinal Test Metrics:")
    for i in range(3):
        min_val, max_val = param_ranges[i]
        
        # Rescale predictions and targets to original range
        pred_rescaled = test_predictions[:, i] * (max_val - min_val) + min_val
        target_rescaled = test_targets[:, i] * (max_val - min_val) + min_val
        
        # Calculate metrics
        mae = torch.mean(torch.abs(pred_rescaled - target_rescaled)).item()
        rmse = torch.sqrt(torch.mean((pred_rescaled - target_rescaled) ** 2)).item()
        
        # R² calculation
        ss_res = torch.sum((target_rescaled - pred_rescaled) ** 2).item()
        ss_tot = torch.sum((target_rescaled - torch.mean(target_rescaled)) ** 2).item()
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        print(f"\nParameter {i+1} ({min_val:.1f}-{max_val:.1f}):")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
    
    # Create final test set visualization
    visualize_predictions(test_predictions, test_targets, param_ranges, monitor.run_dir)
    
    return model, (train_loader, val_loader, test_loader), monitor

# Example usage:
if __name__ == "__main__":

    parameters = np.load("Simulating_Data/20241105_new_labels.npy")

    signals = np.load("Simulating_Data/20241105_new_inputs.npy")

    # Train model with monitoring
    model, loaders, monitor = main(signals, parameters)

    # Access training history
    print(f"Best validation loss: {min(monitor.val_losses)}")
    print(f"Training data saved in: {monitor.run_dir}")


