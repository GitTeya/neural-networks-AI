class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def activate(self, inputs):
        # Ensure the number of inputs and weights are the same
        if len(inputs) != len(self.weights):
            raise ValueError("The number of inputs and weights must be the same")

        # Calculate the weighted sum of inputs
        weighted_sum = sum(input_value * weight for input_value, weight in zip(inputs, self.weights))

        # Apply the threshold
        return 1 if weighted_sum > self.threshold else 0

# Example usage
neuron = McCullochPittsNeuron(weights=[1, -1, 1], threshold=1)
output = neuron.activate([1, 0, 1])
print("Output:", output)


# 定义 McCulloch-Pitts 神经元类
class McCullochPittsNeuron:
    # 初始化函数，接受权重和阈值作为参数
    def __init__(self, weights, threshold):
        self.weights = weights  # 将权重赋值给类的属性
        self.threshold = threshold  # 将阈值赋值给类的属性

    # 激活函数，接受输入值，返回输出结果
    def activate(self, inputs):
        # 检查输入值和权重的数量是否相同
        if len(inputs) != len(self.weights):
            raise ValueError("输入和权重的数量必须相同")

        # 计算输入值和权重的加权和
        weighted_sum = sum(input_value * weight for input_value, weight in zip(inputs, self.weights))

        # 应用阈值，决定输出是1还是0
        return 1 if weighted_sum > self.threshold else 0

# 使用示例
neuron = McCullochPittsNeuron(weights=[1, -1, 1], threshold=1)  # 创建一个神经元实例
output = neuron.activate([1, 0, 1])  # 计算激活函数的输出
print("输出:", output)  # 打印输出结果
