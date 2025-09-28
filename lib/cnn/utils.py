import torch


def save_model_state(model, path='./', validation_id=None):
    """
    save checkpoint (optimizer and model)
    :param validation_id:
    :param model:
    :return:
    """
    path = path+'model_'+str(validation_id)+'.torch'

    print('Saving model: ' + path)

    model = model.module if type(model) is torch.nn.DataParallel else model

    torch.save(model.state_dict(), path)


def load_model_state(model, state_path):
    state = torch.load(state_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state)
