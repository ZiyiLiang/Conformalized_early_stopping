import time
import sys
import os
import torch as th
import numpy as np
import pathlib
from tqdm import tqdm
from torchmetrics.functional import accuracy
from classification import ProbabilityAccumulator as ProbAccum
from scipy.stats.mstats import mquantiles
import pdb

class ConformalizedES:
    def __init__(self, net, device, train_loader, batch_size, max_epoch, learning_rate, criterion, optimizer,
                 val_loader, verbose = True, progress = True):
        self.net = net.to(device)
        self.device = device
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.progress = progress
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.val_loader = val_loader
        self.n_minibatches = len(self.train_loader)

        self.acc = False   # Compute accuracy or not


        if self.verbose:
            print("===== HYPERPARAMETERS =====")
            print("batch_size=", self.batch_size)
            print("n_epochs=", self.max_epoch)
            print("learning_rate=", self.learning_rate)
            print("=" * 30)

    def train_single_epoch(self,epoch):
        """
        Train the model for a single epoch

        :return
        """  
        print_every = self.n_minibatches // 10
        start_time = time.time()

        running_loss = 0.0
        total_train_loss = 0

        # Compute the accuracy for multi-class classification
        if self.acc:
            running_acc = 0.0
            total_train_acc = 0

        for i, (inputs, targets) in enumerate(self.train_loader):

            # Move tensors to correct device
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.criterion(outputs, inputs, targets)
            loss.backward()
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()
            total_train_loss += loss.item()

            if self.acc:
                running_acc += float(accuracy(outputs,targets).cpu().numpy())
                total_train_acc += float(accuracy(outputs,targets).cpu().numpy())

            # print every 10th of epoch
            printing_epochs = np.arange(print_every-1, self.n_minibatches, print_every)
            if self.verbose:
                if i in printing_epochs:
                    if self.acc:
                        print("Epoch {} of {}, {:d}% \t train_loss: {:.2f} train_acc: {:.2f}% took: {:.2f}s".format(
                            epoch + 1, self.max_epoch, int(100 * (i + 1) / self.n_minibatches), running_loss / print_every,
                            100 * (running_acc / print_every), time.time() - start_time)) 
                        running_acc = 0.0
                    else:
                        print("Epoch {} of {}, {:d}% \t train_loss: {:.2f}  took: {:.2f}s".format(
                        epoch + 1, self.max_epoch, int(100 * (i + 1) / self.n_minibatches), 
                        running_loss / print_every, time.time() - start_time)) 
                    
                    running_loss = 0.0
                    start_time = time.time()

        # Return the average training loss and accuracy
        avg_train_loss = total_train_loss / len(self.train_loader)

        if self.acc:
            avg_train_acc = 100 * (total_train_acc / len(self.train_loader))
        
        return (avg_train_loss, avg_train_acc) if self.acc else avg_train_loss


    def full_train(self, save_dir = './models', save_every = 1):
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Save the loss history and accuracy history for plotting
        self.train_loss_history = []
        self.val_loss_history = []
        self.total_val_loss_history = []
        
        if self.acc:
            self.train_acc_history = [] 
            self.val_acc_history = []

        self.model_list = []

        # Save snapshots of the model regularly
        saving_epochs = np.arange(save_every-1, self.max_epoch, save_every)
        for saving_epoch in saving_epochs:
            for i in range(save_every):
                epoch = saving_epoch - save_every + 1 + i
                if self.acc:
                    train_loss, train_acc = self.train_single_epoch(epoch)
                else:
                    train_loss = self.train_single_epoch(epoch)

            snapshot_path = os.path.join(save_dir, 'model'+str(saving_epoch + 1)+'.pth')
            self.model_list.append(snapshot_path)
            th.save(self.net.state_dict(), snapshot_path)

            self.train_loss_history.append(train_loss)
            if self.acc:
                self.train_acc_history.append(train_acc)

            # Evaluate the loss and accuracy on the validation set
            total_val_loss = 0  
            if self.acc:
                total_val_acc = 0

            self.net.eval()    
            for inputs, targets in self.val_loader:
                val_loss = self.get_loss(inputs, targets)
                total_val_loss += val_loss.item()

                if self.acc:
                    val_acc = self.get_acc(inputs, targets)
                    total_val_acc += val_acc

            avg_val_loss = total_val_loss / len(self.val_loader)
            self.val_loss_history.append(avg_val_loss)
            self.total_val_loss_history.append(total_val_loss)
            
            if self.acc:
                avg_val_acc = 100 * (total_val_acc / len(self.val_loader))
                self.val_acc_history.append(avg_val_acc)

            if self.verbose:
                if self.acc:
                    print("val_loss = {:.2f} val_acc = {:.2f}%".format(avg_val_loss, avg_val_acc))
                else:
                    print("val_loss = {:.2f}".format(avg_val_loss))
                print('Snapshot saved at epoch {}.'.format(saving_epoch + 1))
        if self.verbose:
            print('Training done! A list of {} models saved.'.format(len(self.model_list)))

        return None

    def get_acc(self, inputs, targets):
        self.net.eval()
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        with th.no_grad():
            outputs = self.net(inputs)
            acc = float(accuracy(outputs,targets,top_k=1).cpu().numpy())

        return acc
    
    def get_loss(self, inputs, targets):
        self.net.eval()
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        with th.no_grad():
            outputs = self.net(inputs)
            loss = self.criterion(outputs, inputs, targets)

        return loss
    
    def predict(self, inputs):
        self.net.eval()
        inputs = inputs.to(self.device)

        with th.no_grad():
            outputs = self.net(inputs)
        return outputs

class CES_oneClass(ConformalizedES):
    def __init__(self, net, device, train_loader, batch_size, max_epoch, learning_rate, criterion, optimizer,
                val_loader, verbose=True, progress = True):
        super().__init__(net, device, train_loader, batch_size, max_epoch, learning_rate, criterion, optimizer,
                         val_loader, verbose, progress)
        
        # No need to compute accuracy
        self.acc = False

    def select_model(self, test_inputs = None):
        """ Return the indices of the best models from the model list for any test set
        """
        assert self.model_list, 'Call training function to get the model list first!'

        # Return non-conformal benchmark model if no test data is given
        if test_inputs is None:
            best_loss = np.min(self.val_loss_history)
            best_model = self.model_list[np.argmin(self.val_loss_history)]
            test_val_loss_history = self.val_loss_history

        else:
            n_test = len(test_inputs)
            n_val = len(self.val_loader.sampler)
            best_model = [None] * n_test
            best_loss = np.repeat(np.inf, n_test)
            test_val_loss_history = [[] for i in range(n_test)]
            label = th.zeros(1)

            # Load the saved models
            for model_idx, model_path in enumerate(self.model_list):
                self.net.load_state_dict(th.load(model_path))
                # Get the loss for each test point
                val_loss = self.val_loss_history[model_idx]
                for test_idx, test_input in enumerate(tqdm(test_inputs)):
                    test_loss = self.get_loss(test_input[None], label)
                    test_val_loss = val_loss * n_val + test_loss
                    test_val_loss_history[test_idx].append(test_val_loss / (n_val + 1))

                    # Update the best loss and best model
                    if test_val_loss < best_loss[test_idx]:
                        best_loss[test_idx] = test_val_loss
                        best_model[test_idx] = model_path

        return best_loss, best_model, test_val_loss_history



    



class CES_multiClass(ConformalizedES):
    def __init__(self, net, device, train_loader, n_classes, batch_size, max_epoch, learning_rate, criterion,
                 optimizer, val_loader, verbose=True, progress = True, random_state = 2023):
        super().__init__(net, device, train_loader, batch_size, max_epoch, learning_rate, criterion, optimizer,
                         val_loader, verbose, progress)
        
        # Compute accuracy along with the loss
        self.acc = True
        self.n_classes = n_classes
        self.random_state = random_state

    def select_model(self, test_inputs = None):
        """ Return the indices of the best models from the model list for any test set
        """
        assert self.model_list is not None, 'Call training function to get the model list first!'

        # Return non-conformal benchmark model if no test data is given
        if test_inputs is None:
            best_loss = np.min(self.val_loss_history)
            best_model = self.model_list[np.argmin(self.val_loss_history)]
            test_val_loss_history = self.val_loss_history

        else: 
            n_test = len(test_inputs)
            n_val = len(self.val_loader.sampler)
            n_model = len(self.model_list)
            best_model = [[None] * self.n_classes] * n_test
            best_loss = np.full((n_test, self.n_classes), np.inf)
            test_val_loss_history = np.full((n_test, self.n_classes, n_model), -1.0)

            # Load the saved models
            for model_idx, model_path in enumerate(self.model_list):
                print("Loading model {:d} of {:d}...".format(model_idx, len(self.model_list)))
                sys.stdout.flush()
                self.net.load_state_dict(th.load(model_path))
                # Get the loss for each test point
                val_loss = self.val_loss_history[model_idx]
                for test_idx, test_input in enumerate(tqdm(test_inputs)):
                    for label in range(self.n_classes):
                        test_loss = self.get_loss(test_input[None], th.full((1,), label))
                        test_val_loss = (val_loss * n_val + test_loss.item()) / (n_val + 1)

                        # Update the validation test loss history
                        test_val_loss_history[test_idx][label][model_idx] = test_val_loss

                        # Update the best loss and best model for the specific label
                        if test_val_loss < best_loss[test_idx][label]:
                            best_loss[test_idx][label] = test_val_loss
                            best_model[test_idx][label] = model_path

        return best_loss, best_model, test_val_loss_history



class CES_regression(ConformalizedES):
    def __init__(self, net, device, train_loader, batch_size, max_epoch, learning_rate, criterion, optimizer,
                val_loader, verbose=True, progress = True):
        super().__init__(net, device, train_loader, batch_size, max_epoch, learning_rate, criterion, optimizer,
                         val_loader, verbose, progress)
    
    def reg_loss(self, y, y_pred, val_loss):
        # define the quadratic loss for mean regression model
        return y**2 - 2*y*y_pred + y_pred**2 + val_loss

    def find_intersections(self, test_inputs): 
        """Return a list of dictionary containing information of the intersection points
        Args:
            test_inputs (tensor): test covariates

        Returns:
            list of dictionaries that contains information of the intersection point : 
            format of the output (Example): [{'mod1': 4, 'mod2':15, 'intersect': 12.78, 'intersect_loss': 41.21}, 
                                            {'mod1': 82, 'mod2':15, 'intersect': 16.9, 'intersect_loss': 91.6}, ...]
        """

        assert self.model_list, 'Call training function to get the model list first!'
        assert test_inputs is not None, 'Input test covariates first!'
        
        # Store intersection points and point predictions for each test point
        intersects = []
        self.preds = []
        
        # Load the saved models
        for model_idx_i, model_path_i in enumerate(self.model_list):
          self.net.load_state_dict(th.load(model_path_i))
          # Get the total validation loss of each model
          val_loss_i = self.total_val_loss_history[model_idx_i]
          # Make prediction using the model
          pred_i = self.predict(test_inputs).item()
          self.preds.append(pred_i)
        
          for model_idx_j, model_path_j in enumerate(self.model_list):
            
            # Skip duplications
            if model_idx_i >= model_idx_j:
              continue

            self.net.load_state_dict(th.load(model_path_j))
            val_loss_j = self.total_val_loss_history[model_idx_j]
            pred_j = self.predict(test_inputs).item()          

            #y = Symbol('y')            
            #intersect = solve(self.reg_loss(y, pred_i, val_loss_i) - self.reg_loss(y, pred_j, val_loss_j), y)[0]
            
            # compute the intersection points using closed-form formula
            if pred_j - pred_i == 0:
                continue 
            intersect = ((pred_j**2 + val_loss_j) - (pred_i**2 + val_loss_i))/(2*( pred_j - pred_i))
            intersect_loss = self.reg_loss(intersect, pred_j, val_loss_j)

            #print((self.reg_loss(intersect, pred_i, val_loss_i) - self.reg_loss(intersect, pred_j, val_loss_j)))
            #assert np.abs(self.reg_loss(intersect, pred_j, val_loss_j) - self.reg_loss(intersect, pred_i, val_loss_i)) <= 1e-10

            intersects.append({'mod1': model_idx_i, 
                               'mod2':model_idx_j,
                               'intersect': intersect,
                               'intersect_loss': intersect_loss})
        
        return intersects 

    def find_knots(self, test_inputs, eps = 0): 
        
        """Return a list of dictionary containing information of the knots points
        Args:
            test_inputs (tensor): test covariates
            eps (float): tolerance for numerical rounding errors

        Returns:
            subset of the intersections: 
            format of the output (Example): [{'mod1': 4, 'mod2':15, 'intersect': 12.78, 'intersect_loss': 41.21}, 
                                            {'mod1': 82, 'mod2':15, 'intersect': 16.9, 'intersect_loss': 91.6}, ...]
        """
        # find all intersections
        intersects = self.find_intersections(test_inputs)

        # create empty lists to store parameters for computing loss function and selected knots
        loss_params = []
        knots = []
        
        for model_idx, model_path in enumerate(self.model_list):
                # load point prediction of each model
                pred = self.preds[model_idx]
                # load total validation loss of each model
                val_loss = self.total_val_loss_history[model_idx]
                # store the parameters 
                loss_params.append([pred, val_loss])
        
        num_intersects = len(intersects)
        for intersect_idx in range(0, num_intersects):
            # load the current intersection point
            curr_intersect = intersects[intersect_idx]
            # imaginary test response at the intersection 
            curr_y = curr_intersect['intersect']
            # model indices of the intersection             
            model_idx_i = curr_intersect['mod1']
            # loss value at the intersection 
            curr_value = curr_intersect['intersect_loss']
            # compute all possible loss values using all models
            all_values_at_y = [self.reg_loss(curr_y, param[0], param[1]) for param in loss_params]

            # compare loss value at intersection with all other possible loss values
            if np.abs(curr_value - min(all_values_at_y))/np.abs(curr_value) <= eps:
                knots.append(curr_intersect)

        return knots




    def select_model(self, test_inputs = None):
        """Return selected best models and their knot intervals
        Args:
            test_inputs (tensor): test covariates

        Returns:
            list of dictionaries that contains information of the best models: 
            format of the output (Example): [{'model': 4, 'knot_lower': 12.5, 'knot_upper':31.5}, 
                                             {'model': 82,'knot_lower': 31.5, 'knot_upper':99.88}, ...]
        """


        assert self.model_list, 'Call training function to get the model list first!'
        
        start_time = time.time()
        
        # store the best models
        best_models = []

        # if burn_out: 
        #   self.model_list = self.model_list[burn_out:]
        #   print('number of models {}'.format(len(self.model_list)))

        # Return non-conformal benchmark model if no test data is given
        if test_inputs is None:
            best_loss = np.min(self.val_loss_history)
            best_model = self.model_list[np.argmin(self.val_loss_history)]
            test_val_loss_history = self.val_loss_history
            print(best_loss)
            print(best_model)
            
            return best_loss, best_model, test_val_loss_history

        else:
          knots = self.find_knots(test_inputs)
          # count the number of model appearances in knots, if model indice only appeared once, it will be the two 'extreme' models.
          # If the model indice appeared more than once, it will be a model appeared in the middle. 
          line_appearance_count = {}
          map_mod_to_knots = {}
          for knot in knots:
            mod_idx_i = knot['mod1']
            if mod_idx_i in line_appearance_count:
              line_appearance_count[mod_idx_i] += 1
            else:
              line_appearance_count[mod_idx_i] = 1
            if mod_idx_i in map_mod_to_knots:
              map_mod_to_knots[mod_idx_i].append(knot)
            else:
              map_mod_to_knots[mod_idx_i] = [knot]

            mod_idx_j = knot['mod2']
            if mod_idx_j in line_appearance_count:
              line_appearance_count[mod_idx_j] += 1
            else:
              line_appearance_count[mod_idx_j] = 1
            if mod_idx_j in map_mod_to_knots:
              map_mod_to_knots[mod_idx_j].append(knot)
            else:
              map_mod_to_knots[mod_idx_j] = [knot]
            
            
          start_or_end = [k for (k,v) in line_appearance_count.items() if v==1]

          if map_mod_to_knots[start_or_end[0]][0]['intersect'] < map_mod_to_knots[start_or_end[1]][0]['intersect']:
            start_idx = start_or_end[0]
            end_idx = start_or_end[1]
          else:
            start_idx = start_or_end[1]
            end_idx = start_or_end[0]
          
          best_models.append({
              'model': self.model_list[start_idx],
              'knot_lower': -np.inf,
              'knot_upper': map_mod_to_knots[start_idx][0]['intersect']
          })

          curr_idx = start_idx
          seen_knots = []
          lower = map_mod_to_knots[start_idx][0]['intersect']
          while True:
            if curr_idx == start_idx:
              knot = map_mod_to_knots[curr_idx][0]
              seen_knots.append(knot)
            else:
              knots = map_mod_to_knots[curr_idx]
              knot = None
              for knot_candidate in knots:
                if knot_candidate not in seen_knots:
                  knot = knot_candidate
              if knot == None:
                break 
              seen_knots.append(knot)

              y = knot['intersect']
              best_models.append({
                'model': self.model_list[curr_idx],
                'knot_lower': lower,
                'knot_upper': y
              })
              lower = y

            next_idx = None
            for idx in [knot['mod1'], knot['mod2']]:
              if idx != curr_idx:
                next_idx = idx 
            curr_idx = next_idx 
          
          knot = map_mod_to_knots[end_idx][0]
          best_models.append({
              'model': self.model_list[end_idx],
              'knot_lower': knot['intersect'],
              'knot_upper': np.inf
          })

          print(f"elapse time (selecting best models):{time.time() - start_time}")

          return best_models

