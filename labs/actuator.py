
class Actuator:
    r'''这里定义 执行器
    '''

    def __init__(self) -> None:
        self.input_history = []
        pass

    def __call__(self, input):
        r'''G=z^-5
        '''
        self.input_history.append(input)
        self.input_history = self.input_history[-10:
                                                ] if self.input_history.__len__()> 10 else self.input_history
        return self.input_history[-6]if self.input_history.__len__()>= 6 else 0
