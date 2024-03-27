from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from ProtoPNet.models._base_classes import Explainable
from ProtoPNet.models.ProtoPNet._model import ProtoPNetBase, _PositiveLinear


class PPNet(ProtoPNetBase, Explainable):
    """
    ProtoPNet model [Chen+18]_ for prototypical image recognition.

    .. [Chen+18] Chaofan Chen et al. "This looks like that: deep learning
        for interpretable image recognition".
        Url: `http://arxiv.org/abs/1806.10574
        <http://arxiv.org/abs/1806.10574>`_

    :param features: features of the used backend architecture,
        responsible for the feature extraction
    :type features: nn.Module
    :param img_shape: shape of the input image
    :type img_shape: (int, int)
    :param prototype_shape: shape of the prototype tensor
    :type prototype_shape: (int, int, int, int)
    :param proto_layer_rf_info:
    :param n_classes: number of classes in the data
    :type n_classes: int
    :param logger:
    :type logger:
    :param class_specific: Defaults to ``True``.
    :type class_specific: bool
    :param init_weights: flag to mark if the weights should be initialized.
        Defaults to ``True``.
    :type init_weights: bool
    :param prototype_activation_function: activation function for the prototype.
        Defaults to ``"log"``.
    :type prototype_activation_function: str | (torch.Tensor) -> torch.Tensor
    :param add_on_layers_type: type of the add-on layers.
        Defaults to ``"bottleneck"``.
    :type add_on_layers_type: str
    :param add_on_layers_activation: activation function for the add-on layers.
        Defaults to ``"A"``.
    :type add_on_layers_activation: str
    :param positive_weights_in_classifier: flag to mark if the weights
        in the prediction layer should be positive. Defaults to ``False``.
    :type positive_weights_in_classifier: bool
    """

    _AdditionalInformation = namedtuple(
        "_AdditionalInformation", ["min_distances", "distances", "min_indices"]
    )

    def __init__(
        self,
        features,
        img_shape,
        prototype_shape,
        proto_layer_rf_info,
        n_classes,
        logger,
        class_specific=True,
        init_weights=True,
        prototype_activation_function="log",
        add_on_layers_type="bottleneck",
        add_on_layers_activation="A",
        positive_weights_in_classifier=False,
    ):
        super(PPNet, self).__init__(
            features,
            img_shape,
            prototype_shape,
            n_classes,
            logger,
            add_on_layers_type,
            add_on_layers_activation,
        )
        self.__class_specific = class_specific
        self.__n_prototypes = prototype_shape[0]

        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        # Here we are initializing the class identities of the prototypes
        # Without domain specific knowledge we allocate the same number of
        # prototypes for each class

        assert self.__n_prototypes % self._n_classes == 0
        # onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(
            self.__n_prototypes, self._n_classes
        )

        num_prototypes_per_class = self.__n_prototypes // self._n_classes
        for j in range(self.__n_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        self.prototype_vectors = nn.Parameter(
            torch.rand(self.prototype_shape), requires_grad=True
        )

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        if positive_weights_in_classifier:
            self.last_layer = _PositiveLinear(
                self.__n_prototypes, self._n_classes, bias=False
            )
        else:
            self.last_layer = nn.Linear(
                self.__n_prototypes, self._n_classes, bias=False
            )  # do not use bias

        if init_weights:
            self._initialize_weights()

        self.logger.create_csv_log(
            "train_model",
            ("fold", "epoch", "phase"),
            "time",
            "cross entropy",
            "cluster_loss",
            "separation_loss",
            "accuracy",
            "micro_f1",
            "macro_f1",
            "l1",
            "prototype_distances",
            exist_ok=True,
        )

    @property
    def class_specific(self):
        """
        Returns if class specific losses are included in the model loss.

        :return:
        :rtype: bool
        """
        return self.__class_specific

    @property
    def n_prototypes(self):
        """
        Returns the number of prototypes in the model.

        :return: number of prototypes
        :rtype: int
        """
        return self.__n_prototypes

    def conv_features(self, x):
        """
        The feature input to prototype layer.

        :param x: input
        :type x: torch.Tensor
        :return: feature
        :rtype: torch.Tensor
        """
        x = self.features(x)
        x = self.add_on_layers(x)
        return x

    @staticmethod
    def _weighted_l2_convolution(input_, filter_, weights):
        """
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        """
        input2 = input_**2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter_**2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter_ * weights
        weighted_inner_product = F.conv2d(input=input_, weight=weighted_filter)

        # use broadcast
        intermediate_result = (
            -2 * weighted_inner_product + filter_weighted_norm2_reshape
        )
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution(self, x, detach_prototypes=False):
        """
        Apply self.prototype_vectors as l2-convolution filters on input x.

        :param x: features
        :type x: torch.Tensor
        :param detach_prototypes: marks if prototypes should be detached or not.
            Defaults to ``False``.
        :type detach_prototypes: bool
        :return: distances
        :rtype: torch.Tensor
        """
        x2 = x**2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p = (
            self.prototype_vectors.detach()
            if detach_prototypes
            else self.prototype_vectors
        )
        p2 = p**2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=p)
        intermediate_result = -2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape

        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def prototype_min_distances(self, x, detach_prototypes=False):
        """
        Define the additional information about the prototypes of the network.

        :param x: the raw input
        :type x: torch.Tensor
        :param detach_prototypes: marks if prototypes should be detached or not.
            Defaults to ``False``.
        :type detach_prototypes: bool
        :return: minimum distances
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        conv_features = self.conv_features(x)
        distances = self._l2_convolution(conv_features, detach_prototypes)
        # global min pooling
        min_distances, min_indices = F.max_pool2d(
            -distances,
            kernel_size=(distances.size()[2], distances.size()[3]),
            return_indices=True,
        )
        min_distances = -min_distances.view(-1, self.__n_prototypes)
        return min_distances, distances, min_indices

    def distance_2_similarity(self, distances):
        """
        Compute the similarity of prototypes based on their distances
        in the latent space.

        :param distances: distances between the prototypes
        :type distances: torch.Tensor
        :return: prototypes similarity
        :rtype: torch.Tensor
        """
        if self.prototype_activation_function == "log":
            return torch.log((distances + 1) / (distances + self._epsilon))
        elif self.prototype_activation_function == "linear":
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x):
        """
        Forward pass of the network.

        :param x:
        :type x: torch.Tensor
        :return: result of the network on the given data
        :rtype: tuple[torch.Tensor, tuple]
        """
        min_distances, distances, min_indices = self.prototype_min_distances(x)
        prototype_activations = self.distance_2_similarity(min_distances)
        logits = self.last_layer(prototype_activations)
        return logits, self._AdditionalInformation(
            min_distances=min_distances,
            distances=distances,
            min_indices=min_indices,
        )

    def push_forward(self, x):
        """
        this method is needed for the pushing operation

        :param x: input
        :type x: torch.Tensor
        :return: features and feature distances
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        conv_output = self.conv_features(x)
        distances = self._l2_convolution(conv_output)
        return conv_output, distances

    def prune_prototypes(self, prototypes_to_prune):
        """
        prototypes_to_prune: a list of indices each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        """
        prototypes_to_keep = list(
            set(range(self.__n_prototypes)) - set(prototypes_to_prune)
        )

        self.prototype_vectors = nn.Parameter(
            self.prototype_vectors.data[prototypes_to_keep, ...],
            requires_grad=True,
        )

        self._prototype_shape = tuple(self.prototype_vectors.size())
        self.__n_prototypes = self.prototype_shape[0]

        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are consistent
        self.last_layer.in_features = self.__n_prototypes
        self.last_layer.out_features = self._n_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        # self.ones is nn.Parameter
        self.ones = nn.Parameter(
            self.ones.data[prototypes_to_keep, ...], requires_grad=False
        )
        # self.prototype_class_identity is torch tensor,
        # so it does not need .data access for value update
        self.prototype_class_identity = self.prototype_class_identity[
            prototypes_to_keep, :
        ]

    def __repr__(self):
        """
        String representation of the network.

        :return: string representation of the network
        :rtype: str
        """
        return (
            f"PPNet(\n"
            f"\tfeatures: {self.features},\n"
            f"\timg_shape: {self._image_shape},\n"
            f"\tprototype_shape: {self.prototype_shape},\n"
            f"\tproto_layer_rf_info: {self.proto_layer_rf_info},\n"
            f"\tnum_classes: {self._n_classes},\n"
            f"\tepsilon: {self._epsilon}\n)\n"
            f"{super(PPNet, self).__repr__()}"
        )

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        """
        the incorrect strength will be actual strength if -0.5 then input -0.5
        """
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations
        )

    def _initialize_weights(self):
        """
        Initialize weight of all layers.
        """
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)
