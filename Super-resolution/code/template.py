def set_template(args):
    if args.template == "Dual_CNN":
        args.data_train = '291'
        args.data_test = 'Set5'
        args.model = 'dual_cnn'
        args.lr = 1e-4
        args.batch_size = 64
        args.patch_size = 41
        args.epoch = 10000
        args.weight_decay = 0
        args.scale = 2   # todo modify the scale to train
        args.loss = 'L2'
        args.save = 'Dual_CNN_x{}'.format(args.scale)   # todo modify the path to save models and results
        args.gamma = 1
        args.structure_weight = 0.001    # todo modify the weight of structure loss
        args.detail_weight = 0.01   # todo modify the weight of detail loss

    elif args.template == "Dual_CNN-S":
        args.data_train = '291'
        args.data_test = 'Set5'
        args.model = 'dual_cnn'
        args.lr = 1e-4
        args.batch_size = 64
        args.patch_size = 41
        args.epoch = 10000
        args.weight_decay = 0
        args.scale = 2  # todo modify the scale to train
        args.loss = 'L2'
        args.save = 'Dual_CNN-S_x{}'.format(args.scale)  # todo modify the path to save models and results
        args.gamma = 1
        args.structure_weight = 0.01  # todo modify the weight of structure loss
        args.detail_weight = 0  # todo modify the weight of detail loss
