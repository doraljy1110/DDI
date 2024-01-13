import torch
from torch import nn
from SchNet import SchNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class myModel_graph_sch_cnn(nn.Module):
    def __init__(self, num_class=2, cutoff=10.0, num_layers=6, hidden_channels=128,
                 num_filters=128, num_gaussians=50, g_out_channels=5):
        super(myModel_graph_sch_cnn, self).__init__()
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians
        self.model1 = SchNet(energy_and_force=False, cutoff=self.cutoff, num_layers=self.num_layers,
                             hidden_channels=self.hidden_channels, num_filters=self.num_filters,
                             num_gaussians=self.num_gaussians,
                             out_channels=g_out_channels)
        self.model2 = SchNet(energy_and_force=False, cutoff=self.cutoff, num_layers=self.num_layers,
                             hidden_channels=self.hidden_channels, num_filters=self.num_filters,
                             num_gaussians=self.num_gaussians,
                             out_channels=g_out_channels)

        self.fc1 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(32, 32 * 2, bias=True),
            nn.PReLU(),
            nn.Linear(32 * 2, 32, bias=True)
        )
        self.fc2 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(32, 32 * 2, bias=True),
            nn.PReLU(),
            nn.Linear(32 * 2, 32, bias=True)
        )

        self.cnn = CNN_g(in_channel=2, out_channel=num_class)

    def forward1(self, batch_data):
        batch_data.pos = batch_data.pos1
        batch_data.z = batch_data.z1
        batch_data.batch = batch_data.pos1_batch
        self.pred1 = self.model1(batch_data)
        batch_data.pos = batch_data.pos2
        batch_data.z = batch_data.z2
        batch_data.batch = batch_data.pos2_batch
        self.pred2 = self.model2(batch_data)

        self.pred1 = self.fc1(self.pred1)
        self.pred2 = self.fc2(self.pred2)
        self.pred1 = self.pred1.unsqueeze(1)
        self.pred2 = self.pred2.unsqueeze(1)
        self.pred = torch.cat((self.pred1, self.pred2), 1)
        self.pred = self.cnn(self.pred)
        return self.pred

    def forward(self, pos1, z1, pos1_batch, pos2, z2, pos2_batch):

        self.pred1 = self.model1(pos=pos1, z=z1, batch=pos1_batch)
        self.pred2 = self.model2(pos=pos2, z=z2, batch=pos2_batch)

        self.pred1 = self.fc1(self.pred1)
        self.pred2 = self.fc2(self.pred2)
        self.pred1 = self.pred1.unsqueeze(1)
        self.pred2 = self.pred2.unsqueeze(1)
        self.pred = torch.cat((self.pred1, self.pred2), 1)
        self.pred = self.cnn(self.pred)
        return self.pred


class CNN_g(nn.Module):
    def __init__(self, in_channel=2, fc1_hid_dim=256 * 32, out_channel=2, inner_channel=64):
        super(CNN_g, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, inner_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(inner_channel, 128, kernel_size=3, padding=1)
        self.conv31 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv32 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(fc1_hid_dim, inner_channel)
        self.fc2 = nn.Linear(inner_channel, out_channel)
        self.Lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.Lrelu(self.conv1(x))  # batchsize *2 * 32 变为 batchsize *64 * 32
        x = self.Lrelu(self.conv2(x))  # batchsize *128 * 32
        res = x
        x = self.Lrelu(self.conv31(x))  # batchsize *128 * 32
        x = self.Lrelu(self.conv32(x))  # batchsize *128 * 32
        x = res + x  # batchsize *128 * 32
        x = self.Lrelu(self.conv4(x))  # batchsize *256 * 32
        x = self.Lrelu(self.fc1(x.view(x.shape[0], -1)))  # batchsize * 64
        x = self.fc2(x)

        return x



