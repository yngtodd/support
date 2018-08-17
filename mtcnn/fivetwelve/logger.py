def print_progress(epoch, batch_idx, batch_size, train_size, loss):
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


def print_accuracy(epoch, subsite_correct, laterality_correct,
                   behavior_correct, histology_correct, grade_correct, total):
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
    histology = 100 * histology_correct / total
    grade = 100 * grade_correct / total
    print('\nEpoch {:d} Test Accuracy:\nSubsite: {:.2f}\nLaterality: {:.2f}\n' \
          'Behavior: {:.2f}\nGrade: {:.2f}\n'.format(epoch, subsite, laterality, behavior, histology, grade))
