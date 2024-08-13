import torch
import torch.nn as nn
import torch.optim as optim

class LinearRegressionModel(nn.Module):
    def __init__(
        self,
        n_timesteps
    ):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, n_timesteps, bias=False)
        torch.nn.init.ones_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)
    

class SamplingWeights():
    
    def __init__(
        self,
        n_timesteps,
        learning_rate = 10e-4,
    ) -> None:
        
        self.n_timesteps = n_timesteps
        self.weights = LinearRegressionModel(
            n_timesteps=n_timesteps
        )

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.weights.parameters(), lr=learning_rate)
        
    def get_weights(self):
        return self.weights.linear.weight
        
    def result_convert(self, timestep, loss_item):
        B = timestep.shape[0]
        result = torch.zeros((B, self.n_timesteps, 1))
        for i in range(B):
            result[i][timestep[i]] = loss_item[i]
            
        print(result)
        return result
    
    def train_one_epoch(self, timestep, loss_item):
        B = timestep.shape[0]
        y = self.result_convert(timestep, loss_item)
        
        for i in range(B):
            outputs = self.weights.linear.weight
            # print(outputs.shape, y[i].shape)
            loss = self.criterion(outputs, y[i])

            # 기울기 초기화, 역전파, 옵티마이저 스텝
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        
if __name__ == "__main__":
    S = SamplingWeights(n_timesteps=10)
    B = 64
    t = torch.randint(0, 10, (B, ))
    v = torch.randint(0, 10, (B, 1))
    S.train_one_epoch(t, v)
    print(S.get_weights())