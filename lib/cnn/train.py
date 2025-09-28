import torch
import torch.optim as optimizer
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from lib.cnn.predict import predict
from lib.evaluation import evaluate
from lib.cnn.utils import save_model_state
from lib.metrics import ValidationAccuracyMultipleBySpecies
from lib.metrics import ValidationAccuracyMultiple


def fit(model, train, validation, iterations=(90, 130, 150, 170, 180),  log_modulo=500, val_modulo=5, loss=torch.nn.CrossEntropyLoss(),
        lr=0.1,  gamma=0.1,  momentum=0.9, batch_size=124,  n_workers=8,  validation_size=1000,
        metrics=(ValidationAccuracyMultipleBySpecies([1, 10, 30]), ValidationAccuracyMultiple([1, 10, 30])), device_id=0, output_path='./'):
    """
    This function performs a model training procedure.
    :param model: the model that should be trained
    :param train: The training set
    :param validation: The validation set
    :param iterations: This is a list of epochs numbers, it indicates when to changes learning rate
    :param log_modulo: Indicates after how many batches the loss is printed
    :param val_modulo: Indicates after how many epochs should be done a validation
    :param lr: The learning rate
    :param gamma: The coefficient to apply when decreasing the learning rate
    :param momentum: The momentum
    :param batch_size: The mini batch size
    :param n_workers: The number of parallel job for input preparation
    :param validation_size: The maximum number of occurrences to use in validation
    :param metrics: The list of evaluation metrics to use in validation
    """

    # training parameters
    max_iterations = iterations[-1]
    train_loader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=batch_size,  num_workers=n_workers)

    print('Training...')

    opt = optimizer.SGD(model.parameters(), lr=lr, momentum=momentum)

    scheduler = MultiStepLR(opt, milestones=list(iterations), gamma=gamma)

    # number of batches in the ml
    epoch_size = len(train_loader)

    # one log per epoch if value is -1
    log_modulo = epoch_size if log_modulo == -1 else log_modulo

    for epoch in range(max_iterations):
        model.train()

        # printing new epoch
        print('-' * 5 + ' Epoch ' + str(epoch + 1) + '/' + str(max_iterations) +
              ' (lr: ' + str(scheduler.get_lr()) + ') ' + '-' * 5)

        running_loss = 0.0

        for idx, data in enumerate(train_loader):

            # get the inputs
            inputs, labels = data

            # wrap labels in Variable as input is managed through a decorator
            inputs = Variable(inputs.cuda(device_id)) if torch.cuda.is_available() else Variable(inputs)
            labels = Variable(labels.cuda(device_id)) if torch.cuda.is_available() else Variable(labels)

            # zero the parameter gradients
            opt.zero_grad()
            outputs = model(inputs)
            loss_value = loss(outputs, labels)
            loss_value.backward()

            opt.step()

            # print loss
            running_loss += loss_value.item()
            if idx % log_modulo == log_modulo - 1:  # print every log_modulo mini-batches
                print('[%d, %5d] loss: %.5f' % (epoch + 1, idx + 1, running_loss / log_modulo))
                running_loss = 0.0

        # end of epoch update of learning rate scheduler
        scheduler.step(epoch + 1)

        # validation of the model
        if epoch % val_modulo == val_modulo - 1:
            validation_id = str(int((epoch + 1) / val_modulo))

            # predict
            predictions, labels = predict(
                model, validation, batch_size=batch_size, validation_size=validation_size, n_workers=n_workers, device_id=device_id
            )

            # evatuate
            res = '\n[validation_id:' + validation_id + ']\n' + evaluate(predictions, labels, metrics)

            print(res)

            # save model state dict
            # this is not a checkpoint, the model saved can be load but not to restart learning from there
            # this is for the model selection at the end of training for the final evaluation
            save_model_state(model, path=output_path, validation_id=str(validation_id))

    save_model_state(model, validation_id="final")
    # final validation
    print('Final validation: ')

    predictions, labels = predict(model, validation, batch_size=batch_size, validation_size=-1, n_workers=n_workers)

    res = evaluate(predictions, labels, metrics, final=True)

    print(res)

    return predictions
