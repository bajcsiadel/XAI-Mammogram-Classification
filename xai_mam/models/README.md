# Models

In this folder we can find the implemented XAI models trainable for breast lesion classification.

## Structure of a model

Each model consists of two parts:
1. The corresponding configurations in `ProtoPNet/conf/model/`
2. The actual implementation in `ProtoPNet/models/<model_name>` with the 

### Configuration

In the following we are describing how to specify the configuration of a model located in `ProtoPNet/conf`.

In order to validate the different type of parameters in the configuration 
we should specify for each group a `_target_` class (inside group `module`)

E.g. backbone network of ProtoPNet.
```yaml
network:
  _target_: xai_mam.models.xai_mam.config.ProtoPNetBackboneNetwork
  add_on_layer_properties:
    _target_: xai_mam.models.xai_mam.config.AddOnLayerProperties
    type: bottleneck  # regular | bottleneck
    activation: A     # A | B
```

To configure a model first

```tree
conf
├── ...
├── model
│   ├── backbone_only               - has the only responsibility to mark if only the backbone should be trained
│   │   └── yes.yaml
│   ├── network
│   │   ├── _network_config.yaml
│   │   └── ...
│   ├── phases
│   │   ├── <model_name>            - the files in this folder contain model (backbone or explainable) 
│   │   │   │                         specific parameters like `phases`, specific `network` parameters, etc.
│   │   │   ├── backbone.yaml
│   │   │   └── explainable.yaml
│   │   └── ...
│   ├── <model_name>.yaml           - contains the general configurations of the backbone and explainable models
│   └── ...
└── ...
```

If needed new `network` can be added by creating a new file in `model/network`. 
Each network must include as default `_network_config`, containing 
general information about the networks.

The `model/backcbone_only` has the only responsibility to mark if only 
the backbone should be trained. It is necessary because we want to access 
its value in the defaults to decide which file to include from `model/phases/<model_name>`.
In commandline it should be specified as `model.backbone_only@model=yes`

### Code

```tree
├── <model_name>
│   ├── _helpers              - contains helper functions
│   │   ├── __init__.py
│   │   └── ...
│   ├── _model                - contains the classes representing the XAI model
│   │   ├── __init__.py       - could contain a super class for both backbone and explainable models
│   │   ├── backbone.py       - contains the class representing the equivalent backbone model of the XAI model. 
│   │   │                       Must be inherited from `ProtoPNet.models._base_classes.Backbone`
│   │   ├── explainable.py    - contains the class representing the XAI model.
│   │   │                       Must be inherited from `ProtoPNet.models._base_classes.Explainable`
│   │   └── ...
│   ├── _trainer              - containes classes responsible for training a model. 
│   │                           The trainer classes must inherit from `ProtoPNet.models._base_classes.BaseTrainer`
│   │   ├── __init__.py       - could contain a super class for both backbone and explainable trainers
│   │   ├── backbone.py       - contains class responsible to train a backbone model
│   │   └── explainable.py    - contains class responsible to train an explainable model
│   │
│   ├── scripts               - contains scripts that should be executed on trained models
│   │   └── ...
│   ├── __init__.py           - must include references to the created classes: backbone model, 
│   │                           explainable model, backbone trainer, explainable trainer, and the 
│   │                           construct methods responsible to create a backbone model/trainer or 
│   │                           explainable model/trainer
│   └── _construct.py         - contains two functions: construct_model and construct_trainer
│                               responsible to create a backbone model/trainer or 
│                               explainable model/trainer respectively
└── ...
```

:exclamation: Note: `forward` method of the model should either return a number, or
if you want to return `list` or `dictionary` you should use `tuple` or `namedtuple` respectively.