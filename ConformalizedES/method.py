import time
import sys
import os
import torch as th
import numpy as np
import pathlib
from tqdm import tqdm
from torchmetrics.functional import accuracy
from scipy.stats.mstats import mquantiles
from collections import OrderedDict
import numdifftools as nd


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

        self.ID = np.random.randint(0, high=2**31)

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
        print_every = np.maximum(1, self.n_minibatches // 10)
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
            if isinstance(loss, list):
              loss = loss[0]

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
        self.sep_val_loss_history = {}

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

            snapshot_path = os.path.join(save_dir, 'id'+str(self.ID)+'_model'+str(saving_epoch + 1)+'.pth')
            self.model_list.append(snapshot_path)
            th.save(self.net.state_dict(), snapshot_path)

            self.train_loss_history.append(train_loss)
            if self.acc:
                self.train_acc_history.append(train_acc)

            # Evaluate the loss and accuracy on the validation set
            total_val_loss = 0
            sep_val_loss = {}

            if self.acc:
                total_val_acc = 0

            self.net.eval()
            for inputs, targets in self.val_loader:
                val_loss = self.get_loss(inputs, targets)

                if isinstance(val_loss, list):
                  total_val_loss += val_loss[0].item()
                  def sum_dict(d1, d2):
                    for key, value in d1.items():
                      d1[key] = value + d2.get(key, 0)
                    return d1  
                  sep_val_loss = sum_dict(val_loss[1], sep_val_loss)

                else:
                  total_val_loss += val_loss.item()

                if self.acc:
                    val_acc = self.get_acc(inputs, targets)
                    total_val_acc += val_acc

            # avg_val_loss = total_val_loss / len(self.val_loader)
            avg_val_loss = total_val_loss / (len(targets) * len(self.val_loader))
            self.val_loss_history.append(avg_val_loss)
            self.total_val_loss_history.append(total_val_loss)

            for q, v in sep_val_loss.items(): 
              if q in self.sep_val_loss_history: 
                self.sep_val_loss_history[q].append(v.item())
              else:
                self.sep_val_loss_history[q] = [v.item()]
            


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

            if isinstance(loss, list):
              loss[1].update((x, y*len(targets)) for x, y in loss[1].items())
              return [loss[0] * len(targets), loss[1]]

        # NOTE. Careful: the variable loss is the loss averaged over all data points in the batch
        # However, we need the total loss, not the average
        loss = loss * len(targets)

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
                        test_loss = self.get_loss(test_input[None], th.full((1,), label, dtype=th.long))
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
        


    def define_parabolas(self, test_inputs):
        """Return a dictionary of paths and loss functions (characterized by parameters of parabola functions)
        Args:
            test_inputs (tensor): test covariates

        Returns:
            Dictionaries that contains information of the model path and loss function:
            format of the output (Example): {"model_1_path":(a1, b1, c1), "model_2_path":(a2, b2, c2), ...},
        """

        assert self.model_list, 'Call training function to get the model list first!'
        assert test_inputs is not None, 'Input test covariates first!'

        # Store loss functions for each model
        modidx_params = {}

        for model_idx, model_path in enumerate(self.model_list):
            self.net.load_state_dict(th.load(model_path))
            # Get the total validation loss of each model
            val_loss = self.total_val_loss_history[model_idx]
            # Make prediction using the model
            pred = self.predict(test_inputs).item()

            def find_params(y_pred, val_loss):
                return(1.0, -2*y_pred, y_pred**2 + val_loss)

            param = find_params(pred, val_loss)
            modidx_params[model_path] = param

        return modidx_params
      
      
    def define_pinball(self, test_inputs):
        """Return a dictionary of paths and the pinball loss functions (characterized by parameters of pinball loss functions)
        Args:
            test_inputs (tensor): test covariates

        Returns:
            Dictionaries that contains information of the model path and loss function:
            format of the output (Example): {"model_1_path":(a1, b1, c1), "model_2_path":(a2, b2, c2), ...},
        """
        assert self.model_list
        assert test_inputs is not None 
        
        # Store loss functions for each model
        modidx_params_lower = {}
        modidx_params_higher = {}

        for model_idx, model_path in enumerate(self.model_list):
            self.net.load_state_dict(th.load(model_path))
            # Get the total validation loss of each model
            assert len(self.net.quantiles)==2, "quantile length mismatch, input two quantiles only"
            val_loss_lower = self.sep_val_loss_history[self.net.quantiles[0]][model_idx]
            val_loss_higher = self.sep_val_loss_history[self.net.quantiles[1]][model_idx]    
            ## predictions using a list of quantiles
            pred_lower = self.predict(test_inputs)[0][0]
            pred_higher = self.predict(test_inputs)[0][1]
            param_lower = (val_loss_lower, pred_lower, self.net.quantiles[0])
            param_higher = (val_loss_higher, pred_higher, self.net.quantiles[1])
            modidx_params_lower[model_path] = param_lower
            modidx_params_higher[model_path] = param_higher

        return [modidx_params_lower, modidx_params_higher]
    


    def eval_parabola(self, x, params):
        a, b, c = params
        parabola_loss = a*x**2+b*x+c
        return parabola_loss

    def eval_pinball(self, x, params):
      val_loss, pred, quantile = params
      if x >= pred: 
        pinball_loss = val_loss + quantile * (x - pred)
      else: 
        pinball_loss = val_loss + (1-quantile) * (pred - x)
      return pinball_loss


    def intersection_parabolas(self, params1, params2):
        """Return the intersection point of two parabolas. In the CES regression case, the answer is either None or unique
        """

        a1, b1, c1 = params1
        a2, b2, c2 = params2

        assert a1 == a2 # only considering a1=a2=1 cases
        if b1 != b2:
            return -(c1-c2)/(b1-b2)
        else:
            return None

    def intersection_pinball(self, params1, params2):
        """Return the intersection point of two pinball loss functions. In the CES regression case, the answer is either None or unique
        """

        val_loss1, pred1, quantile1 = params1
        val_loss2, pred2, quantile2 = params2

        assert quantile1 == quantile2 # only considering models for the same quantile level
        if pred1 != pred2:
          if pred1 > pred2: 
            return (1-quantile1) * pred1 + quantile1 * pred2 + val_loss1 - val_loss2
          else: 
            return (1-quantile1) * pred2 + quantile1 * pred1 + val_loss2 - val_loss1
        else:
            return None    

    def parabola_min_interval(self, params, x1, x2):
        """Return the minimum value of the given parabola within interval (x1, x2)
        """

        a, b, c = params
        if -b/(2*a) < x2 and -b/(2*a) > x1:
            min_val = self.eval_parabola(-b/(2*a), params)
        else:
            min_val = min(self.eval_parabola(x1, params), self.eval_parabola(x2, params))
        return min_val

    def pinball_min_interval(self, params, x1, x2):
        """Return the minimum value of the given pinball loss function within interval (x1, x2)
        """
        val_loss, pred, quantile = params
        if pred < x2 and pred > x1:
            min_val = val_loss
        else:
            min_val = min(self.eval_pinball(x1, params), self.eval_pinball(x2, params))
        return min_val     

    def sign(self, a):
        return "+" if a >= 0 else "-"


    def merge(self, envelope1, envelope2):
        """Merge step in divide and conquer algorithm
        Args:
            envelope1 (OrderedDict): first previous lower envelope
            envelope2 (OrderedDict): second previous lower envelope

        Returns:
            A new lower envelope constructed after merging the two previous lower envelopes, must be of the same type as input
        """
        # extract the break points from two previous envelopes and sort them in order
        bp1 = list(envelope1.keys())
        bp2 = list(envelope2.keys())
        breakpoints = sorted(set(bp1 + bp2))

        # for convenience, remove the last breakpoint (positive infinity) and deal with it at the end
        breakpoints.remove(np.inf)

        # create an empty ordered dictionary for the new (merged) lower envelope
        envelope = OrderedDict()

        # Next, find new lower envelope within each interval constructed by the adjacent breakpoints

        # set index for calling parabolas/pinball functions in each previous envelope
        index1 = 0
        index2 = 0

        # set the initial lower bound
        prev_x = -np.inf
        # iterately use each breakpoint as upper bounds
        for idx, x in enumerate(breakpoints):
            # identify the parabola/pinball functions contributing to previous lower envelope within the interval
            if x in envelope1:
                params1 = envelope1[x]
                index1 += 1
                if x not in envelope2:
                    params2 = envelope2[bp2[index2]]
                else:
                    params2 = envelope2[x]
                    index2 += 1

            else:
                assert x in envelope2
                params2 = envelope2[x]
                index2 += 1
                params1 = envelope1[bp1[index1]]

            # evaluate the two parabolas/pinball functions identified above at the break point and find their intersection point
            if self.method == 'mse':
              val1 = self.eval_parabola(x, params1)
              val2 = self.eval_parabola(x, params2)
              inter_x = self.intersection_parabolas(params1, params2)

              if val1 == val2:
                  val1 = self.parabola_min_interval(params1, prev_x, x)
                  val2 = self.parabola_min_interval(params2, prev_x, x)
            
            elif self.method == 'quantile': 
              val1 = self.eval_pinball(x, params1)
              val2 = self.eval_pinball(x, params2)
              inter_x = self.intersection_pinball(params1, params2)

              if val1 == val2:
                  val1 = self.pinball_min_interval(params1, prev_x, x)
                  val2 = self.pinball_min_interval(params2, prev_x, x)

            # identify which parabola/pinball functions constructs the new lower envelope within this interval
            if inter_x is None:
                if val1 < val2:
                    envelope[x] = params1
                else:
                    envelope[x] = params2
            else:
                if val1 < val2:
                    if inter_x > x or inter_x < prev_x:
                        envelope[x] = params1
                    else:
                        envelope[inter_x] = params2
                        envelope[x] = params1
                else:
                    if inter_x > x or inter_x < prev_x:
                        envelope[x] = params2
                    else:
                        envelope[inter_x] = params1
                        envelope[x] = params2

            # update lower bound
            prev_x = x

        # find the new lower envelope for the last interval (largest breakpoint, infinity), or (negative infinity, infinity) if the previous envelop only consists one parabola/pinball function
        params1 = envelope1[np.inf]
        params2 = envelope2[np.inf]
        if self.method == 'mse':
          inter_x = self.intersection_parabolas(params1, params2)
        elif self.method == 'quantile':
          inter_x = self.intersection_pinball(params1, params2)

        if self.method == 'mse':
          if len(breakpoints) >= 1:
              x = breakpoints[-1]
              val1 = self.eval_parabola(x, params1)
              val2 = self.eval_parabola(x, params2)
          else:
              a1, b1, c1 = params1
              a2, b2, c2 = params2
              val1 = (-b1/(2*a1), c1)
              val2 = (-b2/(2*a2), c2)

        elif self.method == 'quantile':
          if len(breakpoints) >= 1:
              x = breakpoints[-1]
              val1 = self.eval_pinball(x, params1)
              val2 = self.eval_pinball(x, params2)
          else:
              val_loss1, pred1, quantile1 = params1
              val_loss2, pred2, quantile2 = params2
              val1 = (pred1, val_loss1)
              val2 = (pred2, val_loss2)
     

        if inter_x is None:
            if val1 < val2:
                envelope[np.inf] = params1
            else:
                envelope[np.inf] = params2
        else:
            if val1 < val2:
                if len(breakpoints) >= 1 and inter_x < x:
                    envelope[np.inf] = params1
                else:
                    envelope[inter_x] = params1
                    envelope[np.inf] = params2
            else:
                if len(breakpoints) >= 1 and inter_x < x:
                    envelope[np.inf] = params2
                else:
                    envelope[inter_x] = params2
                    envelope[np.inf] = params1

        # update the new lower envelope
        # sort merged envelope
        envelope = OrderedDict(sorted(envelope.items()))
        # remove points that are not associated with change of parabolas/pinball functions (breakpoints from previous envelopes)
        seen  = set()
        for key, value in list(envelope.items())[::-1]:
            if value in seen:
                envelope.pop(key)
            else:
                seen.add(value)

        # print("new envelope:", envelope)
        return envelope

    def compute_lower_envelope(self, list_params):
        """Given a list of parabolas/pinball functions, find the lower envelop by divide and conquer algorithm in O(nlogn)
        Args:
            list_params (list): A list of parameters characterizing parabolas/pinball functions (e.g. [(1.0, 0.51, 3), (1.0, 3.11, 2), ...])

        Returns:
            The lower envelope of the given list of parabolas/pinball functions together with its corresponding upper bound of the interval:
            format of the output (Example): OrderedDict([(-0.16, (1.0, 0.97, -0.77)), (-0.12, (1.0, -0.16, -0.96), ...)]),
            meaning that in the interval (neg infinity, -0.16), the parabola/pinball function (1.0, 0.97, -0.77) contributes to the lower envelope, and in the interval (-0.16, -0.12), the parabola/pinball function (1.0, -0.16, -0.96) contributes to the lower envelope
        """
        num_samples = len(list_params)

        if num_samples == 1:
            # single parabola/pinball function is itself the lower envelope
            envelope = OrderedDict()
            envelope[np.inf] = list_params[0]
        elif num_samples == 0:
            envelope = OrderedDict()
        else:
            envelope1 = self.compute_lower_envelope(list_params[:num_samples//2])
            envelope2 = self.compute_lower_envelope(list_params[num_samples//2:])
            envelope = self.merge(envelope1, envelope2)

        return envelope

    
    def select_model(self, test_inputs = None, return_time_elapsed = False, method = 'mse', for_visualization = False):
        """Return selected best models and their associated intervals based on the resulting lower envelope
        Args:
            test_inputs (tensor): test covariates
            return_time_elapsed (Bool): True for returning time elapsed
            methods (string): which loss function to use (choosing between 'mse' and 'quantile', default is 'mse')

        Returns:
            list of dictionaries that contains information of the best models:
            format of the output (Example): [{'model': 4, 'knot_lower': 12.5, 'knot_upper':31.5},
                                             {'model': 82,'knot_lower': 31.5, 'knot_upper':99.88}, ...]
        """


        assert self.model_list, 'Call training function to get the model list first!'
        assert method in ['mse', 'quantile'], 'Choose from "mse" and "quantile"'

        self.method = method

        start_time = time.time()

        # store the best models
        best_models = []
        best_models_lower = []
        best_models_higher = []

        # if burn_out:
        #   self.model_list = self.model_list[burn_out:]
        #   print('number of models {}'.format(len(self.model_list)))

        # Return non-conformal benchmark model if no test data is given
        if test_inputs is None:
            if method == 'mse':
                best_loss = np.min(self.val_loss_history)
                best_model = self.model_list[np.argmin(self.val_loss_history)]
                test_val_loss_history = self.val_loss_history
                # print(best_loss)
                # print(best_model)

                return best_loss, best_model, test_val_loss_history
            elif method == 'quantile':
                val_loss_history_lower = self.sep_val_loss_history[self.net.quantiles[0]]
                val_loss_history_higher = self.sep_val_loss_history[self.net.quantiles[1]]
                best_loss_lower = np.min(self.sep_val_loss_history[self.net.quantiles[0]])
                best_model_lower = self.model_list[np.argmin(self.sep_val_loss_history[self.net.quantiles[0]])]
                best_loss_higher = np.min(self.sep_val_loss_history[self.net.quantiles[1]])
                best_model_higher = self.model_list[np.argmin(self.sep_val_loss_history[self.net.quantiles[1]])]

                return [best_loss_lower, best_model_lower, val_loss_history_lower, best_loss_higher, best_model_higher, val_loss_history_higher]




        else:
            # for each of the saved model, define their associated loss functions and record model paths
            if self.method == "mse":
                modidx_params = self.define_parabolas(test_inputs)
                params = list(modidx_params.values())
                # find the lower envelope of the parabolas
                lower_envelope = self.compute_lower_envelope(params)

                # identify the best models with corresponding intervals using the lower envelope
                lower = -np.inf
                for key,value in lower_envelope.items():
                    mod_idx = [k for k, v in modidx_params.items() if v == value]
                    upper = key
                    best_models.append({
                        'model':mod_idx[0],
                        'knot_lower':lower,
                        'knot_upper':upper})
                    lower = upper
                
                time_elapsed = time.time() - start_time
                # print(f"elapse time (selecting best models using old method):{time_elapsed}")
                if return_time_elapsed:
                    return best_models, time_elapsed
                if for_visualization:
                    return params, lower_envelope, best_models

                return best_models
            
            elif self.method == "quantile":
                modidx_params_lower, modidx_params_higher = self.define_pinball(test_inputs)
                params_lower = list(modidx_params_lower.values())
                params_higher = list(modidx_params_higher.values())
                # find the lower envelope of the parabolas
                lower_envelope_lower = self.compute_lower_envelope(params_lower)
                lower_envelope_higher = self.compute_lower_envelope(params_higher)

                # identify the best models with corresponding intervals using the lower envelope
                lower = -np.inf
                for key,value in lower_envelope_lower.items():
                    mod_idx = [k for k, v in modidx_params_lower.items() if v == value]
                    upper = key
                    best_models_lower.append({
                        'model':mod_idx[0],
                        'knot_lower':lower,
                        'knot_upper':upper})
                    lower = upper

                # identify the best models with corresponding intervals using the lower envelope
                lower = -np.inf
                for key,value in lower_envelope_higher.items():
                    mod_idx = [k for k, v in modidx_params_higher.items() if v == value]
                    upper = key
                    best_models_higher.append({
                        'model':mod_idx[0],
                        'knot_lower':lower,
                        'knot_upper':upper})
                    lower = upper


                time_elapsed = time.time() - start_time
                # print(f"elapse time (selecting best models using old method):{time_elapsed}")
                if return_time_elapsed:
                    return best_models_lower, best_models_higher, time_elapsed
                if for_visualization:
                    return {"lower_quantile":[params_lower,lower_envelope_lower, best_models_lower],
                        "higher_quantile":[params_lower,lower_envelope_lower, best_models_higher],}

                return best_models_lower, best_models_higher




