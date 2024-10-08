import datetime

import torch

from datasets import *
from module import *
from params import elements

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
batch = 12800

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train(data, label, model, loss_fn, optimizer, epoch):
    size = len(label)
    model.train()
    for i in range(0, size, batch):

        X = data[i: i + batch, :]
        y = label[i: i + batch]
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        pred = torch.squeeze(pred)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            loss, current = loss.item(), i
            print(f"\rEpoch: {epoch + 1:d} Batch: [{current:>5d}/{size:>5d}] train_loss: {loss:>7f} lr: {optimizer.state_dict()['param_groups'][0]['lr']}",
                  end='', flush=True)

def test(data, label, model, loss_fn):
    size = len(label)
    num_batches = int(size / batch)
    model.eval()
    test_loss, correct, depth_loss = 0, 0, 0
    with torch.no_grad():
        for i in range(0, size, batch):
            X = data[i: i + batch, :]
            y = label[i: i + batch]
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred = torch.squeeze(pred)
            test_loss += loss_fn(pred, y).item()

            # vis = pred[:, 0]
            # depth = pred[:, 1]
            #
            # vis[vis > 0.5] = 1.0
            # vis[vis < 0.5] = 0.0
            #
            # result = torch.eq(vis, y[:, 0])
            # depth_loss += loss_fn(depth, y[:, 1]).item()

            # correct += result.sum().item()

    test_loss /= num_batches
    depth_loss /= num_batches
    correct /= size
    print(f"\nTest Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} , depth loss: {depth_loss:>8f} \n", flush=True)

    return test_loss

if __name__ == '__main__':
    setup_seed(19990201)

    instance_name = elements[0]
    building_ID = ""
    file_name = "instance_repository"
    NN_type = "vis"
    model_depth = 4
    layer_size = 256

    print(f"Learning instance {instance_name}!")
    origin_path = f"../../geometry_data/{file_name}/large/origin{instance_name}.exr"
    direction_path = f"../../geometry_data/{file_name}/large/direction{instance_name}.exr"
    data, label = loadNormalizedDatasetsBalanceVIS(origin_path, direction_path) if NN_type == "vis" \
        else loadNormalizedDatasetsDepth(origin_path, direction_path)

    data_size = 3 if NN_type == "vis" else 4
    for i in range(data_size):
        origin_path = f"../../geometry_data/{file_name}/large/origin{instance_name}{i + 1}.exr"
        direction_path = f"../../geometry_data/{file_name}/large/direction{instance_name}{i + 1}.exr"

        # origin_path = f"./geometry_data/{file_name}/large/origin{instance_name}{i + 1}.exr"
        # direction_path = f"./geometry_data/{file_name}/large/direction{instance_name}{i + 1}.exr"

        temp_data, temp_label = loadNormalizedDatasetsBalanceVIS(origin_path, direction_path) if NN_type == "vis" \
            else loadNormalizedDatasetsDepth(origin_path, direction_path)
        data = torch.cat([data, temp_data], dim=0)
        label = torch.cat([label, temp_label], dim=0)

    train_data, train_label, test_data, test_label = getDatasets(data, label)

    # origin_path_prefix = "./geometry_data/originNewTransformGeo"
    # direction_path_prefix = "./geometry_data/directionNewTransformGeo"
    # data, label = loadMultiDatasets(origin_path_prefix, direction_path_prefix, 4)
    # train_data, train_label, test_data, test_label = getDatasets(data, label)

    model_name = f"NeuralVisNetworkWith{model_depth}Res{layer_size}SingleOutput_{NN_type}" if NN_type == "vis" \
        else f"NeuralVisNetworkWith{model_depth}Res{layer_size}SingleOutput_{NN_type}"

    model = NeuralVisNetworkWith4Res128SingleOutput()
    if(model_depth == 4):
        if (layer_size == 128):
            model = NeuralVisNetworkWith4Res128SingleOutput()
        elif (layer_size == 256):
            model = NeuralVisNetworkWith4Res256SingleOutput()
        elif (layer_size == 512):
            model = NeuralVisNetworkWith4Res512SingleOutput()
    elif(model_depth == 6):
        if (layer_size == 128):
            # model = NeuralVisNetworkWith6Res128SingleOutput()
            exit()
        elif (layer_size == 256):
            model = NeuralVisNetworkWith6Res256SingleOutput()
        elif (layer_size == 512):
            # model = NeuralVisNetworkWith6Res512SingleOutput()
            exit()
    model.to(device)

    model_path = f"./module/{file_name}/{NN_type}/NeuralVisNetworkWith4Res256SingleOutput_vis-2024-04-27-19-04-loss=0.004437-epochs=120-CHEVAL_MARLY-model.pth"
    model.load_state_dict(torch.load(model_path))

    # model = NeuralVisNetworkWithRes().to(device)

    # loss_fn = nn.BCELoss()
    loss_fn = nn.MSELoss() if NN_type == "vis" else nn.L1Loss()

    learn_rate = 5e-4
    if (layer_size == 128):
        learn_rate = 5e-4
    elif (layer_size == 256):
        learn_rate = 5e-4
    elif (layer_size == 512):
        learn_rate = 1e-4
    learn_rate = 1e-5

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, min_lr=0)

    # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
    #                                            verbose=False, threshold=0.01, threshold_mode='rel', cooldown=0,
    #                                            min_lr=0, eps=1e-08)
    epochs = 1000
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_data, train_label, model, loss_fn, optimizer, t)
        test_loss = test(test_data, test_label, model, loss_fn)

        scheduler.step(test_loss)

        if (t % 20 == 0):
            local_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
            torch.save(model.state_dict(),
            f"../../PycharmProjects/vis/module/{file_name}/{NN_type}/{model_name}-{local_time}-loss={test_loss:>5f}-epochs={t}-{instance_name}{building_ID}-model.pth")
            print("Saved PyTorch Model State to model.pth")
        train_data, train_label = shuffleDatasets(train_data, train_label)
    print("Done!")

