import torch
import torch.nn as nn


class MTCNN(nn.Module):
    def __init__(self, wv_matrix, kernel1=3, kernel2=4, kernel3=5, num_filters1=100,
                 num_filters2=100, num_filters3=100, dropout1=0.5, dropout2=0.5, dropout3=0.5,
                 max_sent_len=1500, word_dim=300, vocab_size=2881, subsite_size=8,
                 laterality_size=2, behavior_size=2, grade_size=4, alt_model_type=None):
        super(MTCNN, self).__init__()
        """
        Multi-task CNN model for document classification.

        Parameters:
        ----------
        * `wv_matrix` []
            Word vector matrix

        * `kernel*`: [int]
            Kernel filter size at convolution *.

        * `num_filters*` [int]
            Number of convolutional filters at convolution *.

        * `dropout*`: [float]
            Probability of elements being zeroed at convolution *.

        * `max_sent_len [int]
            Maximum sentence length.

        * `word_dim` [int, default=30]
            Word dimension.

        * `vocab_size`: [int]
            Vocabulary size.

        * `subsite_size`: [int]
            Class size for subsite task.

        * `laterality_size`: [int]
            Class size for laterality task.

        * `behavior_size`: [int]
            Class size for behavior task.

        * `grade_size`: [int]
            Class size for grade task.

        * `alt_model_type`: [str, default=None]
            Alternative type of model being used.
            -Options:
                "static":
                "multichannel":
        """
        self.wv_matrix = wv_matrix
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.kernel3 = kernel3
        self.num_filters1 = num_filters1
        self.num_filters2 = num_filters2
        self.num_filters3 = num_filters3
        self.max_sent_len = max_sent_len
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = dropout3
        self.word_dim = word_dim
        self.vocab_size = vocab_size
        self.subsite_size = subsite_size
        self.laterality_size = laterality_size
        self.behavior_size = behavior_size
        self.grade_size = grade_size
        self.alt_model_type = alt_model_type
        self._filter_sum = None
        self._sum_filters()

        self.embedding = nn.Embedding(self.vocab_size + 2, self.word_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(self.wv_matrix))

        if self.alt_model_type == 'static':
            self.embedding.weight.requires_grad = False
        elif self.alt_model_type == 'multichannel':
            self.embedding2 = nn.Embedding(self.vocab_size + 2, self.word_dim, padding_idx=self.vocab_size + 1)
            self.embedding2.weight.data.copy_(torch.from_numpy(self.wv_matrix))
            self.embedding2.weight.requires_grad = False
            self.IN_CHANNEL = 2

        self.convblock1 = nn.Sequential(
            nn.Conv1d(1, self.num_filters1, self.kernel1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(p=self.dropout1)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv1d(1, self.num_filters2, self.kernel2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(p=self.dropout2)
        )

        self.convblock3 = nn.Sequential(
            nn.Conv1d(1, self.num_filters3, self.kernel3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(p=self.dropout3)
        )

        self.fc1 = nn.Linear(self._filter_sum, self.subsite_size)
        self.fc2 = nn.Linear(self._filter_sum, self.laterality_size)
        self.fc3 = nn.Linear(self._filter_sum, self.behavior_size)
        self.fc4 = nn.Linear(self._filter_sum, self.grade_size)

    def _sum_filters(self):
        """Get the total number of convolutional filters."""
        self._filter_sum = self.num_filters1 + self.num_filters2 + self.num_filters3

    def forward(self, x):
        x = self.embedding(x).view(-1, 1, self.word_dim * self.max_sent_len)
        if self.alt_model_type == "multichannel":
            x2 = self.embedding2(x).view(-1, 1, self.word_dim * self.max_sent_len)
            x = torch.cat((x, x2), 1)

        conv_results = []
        conv_results.append(self.convblock1(x).view(-1, self.num_filters1))
        conv_results.append(self.convblock2(x).view(-1, self.num_filters2))
        conv_results.append(self.convblock3(x).view(-1, self.num_filters3))
        x = torch.cat(conv_results, 1)

        out_subsite = self.fc1(x)
        out_laterality = self.fc2(x)
        out_behavior = self.fc3(x)
        out_grade = self.fc4(x)
        return out_subsite, out_laterality, out_behavior, out_grade
