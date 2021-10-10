class ModelNameGenerator:
    def __init__(self, method_name, model_name_suffix=''):
        super(ModelNameGenerator, self).__init__()
        self.params = [('model', method_name)]
        self.model_name_suffix = model_name_suffix

    def add_epoch_number(self, param_value):
        self.params.append(('en', str(param_value)))
        return self

    def add_batch_size(self, param_value):
        self.params.append(('bs', str(param_value)))
        return self

    def add_learning_rate(self, param_value):
        self.params.append(('lr', str(param_value)))
        return self

    def add_optimizer_name(self, param_value):
        self.params.append(('op', str(param_value)))
        return self

    def add_geometric_feature(self, param_value):
        self.params.append(('geo', str(param_value)))
        return self

    def add_hidden_size(self, param_value):
        self.params.append(('hs', str(param_value)))
        return self

    def add_input_type(self, param_value):
        self.params.append(('it', str(param_value)))
        return self

    def add_momentum(self, param_value):
        self.params.append(('momentum', str(param_value)))
        return self

    def add_weight_decay(self, param_value):
        self.params.append(('wd', str(param_value)))
        return self

    def add_hidden_layers(self, param_value):
        self.params.append(('hl', str(param_value)))
        return self

    def add_dropout(self, param_value):
        self.params.append(('dropout', str(param_value)))
        return self

    def add_split(self, param_value):
        self.params.append(('split', str(param_value)))
        return self

    def add_steps(self, param_value):
        self.params.append(('steps', str(param_value)))
        return self

    def add_action_repetitions(self, param_value):
        self.params.append(('ar', str(param_value)))
        return self

    def add_step_size_lr(self, param_value):
        self.params.append(('stlr', str(param_value)))
        return self

    def add_lambda(self, lbd):
        self.params.append(('lbd', str(lbd)))
        return self

    def add_gamma_step_lr(self, param_value):
        self.params.append(('gammastlr', str(param_value)))
        return self

    def add_is_3d(self, is_3d):
        self.params.append(('', '3D' if is_3d else '2D'))
        return self

    def add_random_rotation_y(self, is_random_rotation_y):
        if is_random_rotation_y:
            self.params.append(('', 'rotations'))
        return self

    def add_is_bias_used(self, is_bias_used):
        if is_bias_used:
            self.params.append(('', 'bias'))
        return self

    def add_is_normalization_used(self, is_normalization_used):
        if is_normalization_used:
            self.params.append(('', 'normalized'))
        return self

    def add_is_tau(self, is_tau):
        if is_tau:
            self.params.append(('', 'tau'))
        return self

    def add_is_two_layers_used(self, is_two_layers_used):
        if is_two_layers_used:
            self.params.append(('', '2layers'))
        return self

    def generate(self):
        if self.model_name_suffix != '':
            self.params.append(('', self.model_name_suffix))
        return '_'.join([i[1] if i[0] == '' else i[0] + '_' + i[1] for i in self.params])
