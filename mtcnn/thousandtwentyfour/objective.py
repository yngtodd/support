def objective(hparams):
    """
    Train the model.

    Parameters:
    ----------
    * `hparams`: [list]
        Hyperparameters set by HyperSpace.

    * `train_loader` [torch.utils.data.Dataloader]
        Data loader to load the test set.

    * `test_loader`: [torch.utils.data.Dataloader]
        Data loader to load the validation set.

    * `optimizer`: [torch.optim optimizer]
        Optimizer for learning the model.

    * `criterion`: [torch loss function]
        Loss function to measure learning.

    * `train_size`: [int]
        Size of the training set (for logging).

    * `args`: [argparse object]
        Parsed arguments.
    """
    optimization = args.optimizer
    kernel1, kernel2, kernel3, num_filters1, num_filters2, num_filters3, dropout1, dropout2, dropout3 = hparams

    kernel1 = int(kernel1)
    kernel2 = int(kernel2)
    kernel3 = int(kernel3)
    num_filters1 = int(num_filters1)
    num_filters2 = int(num_filters2)
    num_filters3 = int(num_filters3)
    dropout1 = float(dropout1)
    dropout2 = float(dropout2)
    dropout3 = float(dropout3)

    global model
    model = MTCNN(
        wv_matrix, kernel1=kernel1, kernel2=kernel2, 
        kernel3=kernel3, num_filters1=num_filters1,
        num_filters2=num_filters2, num_filters3=num_filters3,
        dropout1=dropout1, dropout2=dropout2, dropout3=dropout3
    )

    if args.cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.cuda()

    global optimizer
    if optimization == 0:
        optimizer = optim.Adam(model.parameters())
    elif optimization == 1:
        optimizer = optim.Adadelta(model.parameters())
    elif optimization == 2:
        optimizer = optim.Adagrad(model.parameters())
    elif optimization == 3:
        optimizer = optim.ASGD(model.parameters())
    else:
        optimizer = optim.SGD(model.parameters())

    model.train()
    global epoch
    for epoch in range(1, args.num_epochs + 1):
            for batch_idx, sample in enumerate(train_loader):
                sentence = sample['sentence']
                subsite = sample['subsite']
                laterality = sample['laterality']
                behavior = sample['behavior']
                grade = sample['grade']

                if args.cuda:
                    sentence = sentence.cuda()
                    subsite = subsite.cuda()
                    laterality = laterality.cuda()
                    behavior = behavior.cuda()
                    grade = grade.cuda()
                    if args.half_precision:
                        sentence = sentence.half()
                        subsite = subsite.half()
                        laterality = laterality.half()
                        behavior = behavior.half()
                        grade = grade.half()

                sentence = Variable(sentence)
                subsite = Variable(subsite)
                laterality = Variable(laterality)
                behavior = Variable(behavior)
                grade = Variable(grade)

                optimizer.zero_grad()
                out_subsite, out_laterality, out_behavior, out_grade = model(sentence)
                loss_subsite = criterion(out_subsite, subsite)
                loss_laterality = criterion(out_laterality, laterality)
                loss_behavior = criterion(out_behavior, behavior)
                loss_grade = criterion(out_grade, grade)
                loss = loss_subsite + loss_laterality + loss_behavior + loss_grade
                loss.backward(retain_graph=False)
                optimizer.step()

    ave_loss = validate(test_loader, criterion, args)
    # Clean up
    del sentence, subsite, laterality, behavior, grade
    torch.cuda.empty_cache() 
    return ave_loss
