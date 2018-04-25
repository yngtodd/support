import torch


def print_progress(epoch, batch_idx, batch_size, train_size, loss, logger=None):
    """
    Print the learning progress.

    Parameters:
    ----------
    * `epoch`: [int]
        Epoch number.
    
    * `loss`: [float]
        Loss value.

    * `batch_size`: [int]
        Batch size for training to keep track of progress.

    * `train_size`: [int]
        Size of the training set.
    """
    print('Epoch: {:d}, Step: [{:d}/{:d}], Loss: {:.4f}' \
          .format(epoch, (batch_idx + 1) * batch_size, train_size, loss))

    if logger:
        logger.info('Epoch: {:d}, Step: [{:d}/{:d}], Loss: {:.4f}' \
                    .format(epoch, (batch_idx + 1) * batch_size, train_size, loss))


def print_accuracy(epoch, subsite_correct, laterality_correct, 
                   behavior_correct, grade_correct, total, logger=None):
    """
    Print the accuracy for each task.

    Parameters:
    ----------
    * `subsite_correct`: [int]
        Number of correctly predicted instances for subsite.

    * `laterality_correct: [int]
        Number of correctly predicted instances for laterality.

    * `behavior_correct`: [int]
        Number of correctly predicted instances for behavior.

    * `grade_correct`: [int]
        Number of correctly predicted instances for grade.

    * `total`: [int]
        Number of test cases.
    """
    subsite = 100 * subsite_correct / total
    laterality = 100 * laterality_correct / total
    behavior = 100 * behavior_correct / total
    grade = 100 * grade_correct / total
    print('\nEpoch {:d} Test Accuracy:\nSubsite: {:.2f}\nLaterality: {:.2f}\n' \
          'Behavior: {:.2f}\nGrade: {:.2f}\n'.format(epoch, subsite, laterality, behavior, grade))

    if logger:
        logger.info('\nEpoch {:d} Test Accuracy:\nSubsite: {:.2f}\nLaterality: {:.2f}\n' \
                    'Behavior: {:.2f}\nGrade: {:.2f}\n'.format(epoch, subsite, laterality, behavior, grade))


def fbeta_score(y_true, y_pred, beta=1, threshold=0.5, eps=1e-9):
    beta2 = beta**2

    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    score = torch.mean((precision*recall).div(precision.mul(beta2) + recall + eps).mul(1 + beta2))
    return score


def logf1(epoch, subsitef1, lateralityf1, behaviorf1, gradef1):
    """
    Log F1 score.
    """
    logger.info('\nEpoch {:d} Test F1:\nSubsite: {:.2f}\nLaterality: {:.2f}\n' \
          'Behavior: {:.2f}\nGrade: {:.2f}\n'.format(epoch, subsitef1, lateralityf1, behaviorf1, gradef1))
    
